import cv2
import datetime
import pytz
import time

import pyzed.sl as sl
import numpy as np
import plotly.graph_objects as go

from numba import jit, prange

class Constant():
    def __init__(self):
        # PARAMETROS ALTURA
        self.THRESHOLD_MIN   = 0
        self.TOLVA_MIN       = 1400 #700
        self.THRESHOLD_IDEAL = 2100 #950
        self.TOLVA_MAX       = 2700 #1050
        self.THRESHOLD_MAX   = 3300 #1300
        self.CAMERA_DIST     = 9000 #3000

        # UMBRALES
        self.UMBRAL_MAX       = 0.7
        self.UMBRAL_VOLCADURA = 0.8
        self.UMBRAL_VACIO     = 5000
        self.SEG_MIN          = 700000
        self.SEG_MAX          = 1500000
        self.POS_MAX          = 150

        # DIMENSIONES TOLVA
        self.INIT_VOL = 34
        self.HEIGHT   = self.TOLVA_MIN/1000
        self.WIDTH    = 2.4
        self.LENGTH   = 9
        
        # COLORES
        self.COLOR_BLACK    = (  0,   0,    0)
        self.COLOR_BLUE     = (255,   0,    0)
        self.COLOR_SKYBLUE  = (255, 255,    0)
        self.COLOR_GREEN    = (  0, 255,    0)
        self.COLOR_YELLOW   = (  0, 255,  255)
        self.COLOR_SYELLOW  = (  0, 255,  127)
        self.COLOR_RED      = (  0,   0,  255)
        self.COLOR_WHITE    = (255, 255,  255)

class Cubage(Constant):

    def __init__(self, rotation = (0,0,0), ubication = None, cut_image = (0,0,0,0), name = None):
        super().__init__()
        self.cam, self.init_params = self.setup_camera(ubication)
        self.rotation = self.get_transformation_matrix(rotation).T
        
        self.name = name
        (self.__X, self.__Y, self.__XF, self.__YF) = cut_image
        self.__W = self.__XF - self.__X
        self.__H = self.__YF - self.__Y

        self.status = self.cam.open(self.init_params)
        if self.status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(self.status)+". Exit program.")
            exit(-1)
        
        self.enable_tracking_and_mapping()

        self.__runtime      = sl.RuntimeParameters(confidence_threshold=100, texture_confidence_threshold=100)
        self.__object_image = sl.Mat()
        self.__mat_cloud    = sl.Mat()

        self.__state      = np.zeros((1, 5), dtype=bool)
        self.__middle     = np.zeros((1,5), dtype=np.uint32)
        self.__center     = np.zeros((1,5), dtype=np.uint32)
        
        self.cant = 0
        self.date = None
        self.hour = None
        self.image = np.zeros(( self.cam.get_camera_information().camera_configuration.resolution.height, self.cam.get_camera_information().camera_configuration.resolution.width,3), dtype = np.uint8)
        self.depth_image            = np.zeros((self.__H,self.__W,3), dtype=np.uint8)
        self.mask_tolva_min_to_min  = np.zeros((self.__H,self.__W), dtype = np.bool_)
        self.alpha                  = np.zeros((self.__H,self.__W), dtype=np.uint8)

    def get_transformation_matrix(self, rotation):
        (roll, pitch, yaw) = rotation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def setup_camera(self, ubication):
        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.camera_resolution = sl.RESOLUTION.HD2K
        if ubication is not None:   init_params.set_from_svo_file(ubication)

        return cam, init_params

    def enable_tracking_and_mapping(self):
        tracking_parameters = sl.PositionalTrackingParameters()
        self.cam.enable_positional_tracking(tracking_parameters)

        mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.HIGH,mapping_range =  sl.MAPPING_RANGE.MEDIUM,save_texture = True,use_chunk_only = True,reverse_vertex_order = False,map_type = sl.SPATIAL_MAP_TYPE.MESH)
        self.cam.enable_spatial_mapping(mapping_parameters)
        return 
    
    def UNIXtoDATE(self, epoch_time):
        utc_time = datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc)
        target_time_zone = pytz.timezone('America/Lima')
        target_time = utc_time.astimezone(target_time_zone)

        return target_time.strftime('%Y-%m-%d'), target_time.strftime('%H:%M:%S')

    @staticmethod
    @jit(nopython=True, parallel=True)
    def DOT(points, rotation):
        __H, __W = points.shape[:2]
        result = np.empty((__H, __W, 3), dtype=np.float64)
        
        for i in prange(__H):
            for j in prange(__W):
                for k in prange(3):
                    result[i, j, k] = (
                        points[i, j, 0] * rotation[0, k] +
                        points[i, j, 1] * rotation[1, k] +
                        points[i, j, 2] * rotation[2, k]
                    )
        return result
    
    def Transformation(self, points):     
        points[:,:,2] += self.CAMERA_DIST 
        points = self.DOT(points, self.rotation) #Rotación #points = np.dot(points.reshape(-1, 3), rotation).reshape(__H, __W, 3)
        points[:,:,2] = cv2.GaussianBlur(points[:,:,2], (3, 3), 0) 
        return  points

    def Segmentation(self, points):
        filas, columnas = points.shape
        mask_over_ideal = points >= self.THRESHOLD_IDEAL-300
        
        mask = np.zeros((filas, columnas), dtype=np.uint8)
        mask[mask_over_ideal] = 255

        mask = cv2.dilate(mask, np.ones((2, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        # mask = cv2.dilate(mask, np.ones((2, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  np.ones((3, 3), np.uint8), iterations=1)
        
        ultima_columna = mask[:, -1]
        mask[:, -1][np.argmax(ultima_columna == 255):len(ultima_columna) - np.argmax(ultima_columna[::-1] == 255)] = 255

        primera_columna = mask[:, 0]
        mask[:, 0][np.argmax(primera_columna == 255):len(primera_columna) - np.argmax(primera_columna[::-1] == 255)] = 255
        (contornos,_) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        end_mask = np.zeros((filas, columnas), dtype=np.uint8)
        coords = (self.__X,self.__Y,self.__W,self.__H)

        if contornos:
            max_contorno = max(contornos, key=cv2.contourArea)
            max_area = cv2.contourArea(max_contorno)
            max_contorno  = max_contorno if self.SEG_MAX > max_area > self.SEG_MIN else None

            if max_contorno is not None:
                x, y, w, h = cv2.boundingRect(max_contorno)
                end_mask = cv2.drawContours(end_mask,[max_contorno],-1,255,cv2.FILLED) #end_mask = cv2.rectangle(end_mask, (x,y), (x+w,y+h), 255, -1)
                coords = (x, y, w, h)
        
        return end_mask, coords
  
    def Remove_contour(self, points, mask):
        (contornos,_) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contornos]
        max_index = np.argmax(areas)
        cnt = contornos[max_index]
        
        mask_contour = np.zeros((points.shape[0], points.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_contour, [cnt], -1, 255, 50)
        cv2.dilate(mask_contour, np.ones((2, 9), np.uint8), iterations=5)
                   
        over_contour = points >= (self.THRESHOLD_IDEAL+self.TOLVA_MIN)/2
        mask_over_contour = np.zeros((points.shape[0], points.shape[1]))
        mask_over_contour[over_contour] = points[over_contour] 

        mask_under_ideal = (mask_contour// 255).astype(np.uint8)
        new_points = np.multiply(mask_over_contour, mask_under_ideal)
        high = np.round(np.nanmax(np.mean(new_points[new_points > 0])),-1)

        self.mask_under_ideal = new_points >= self.THRESHOLD_IDEAL
        # print('altura',high)

        self.THRESHOLD_IDEAL = int(high) - 200  #3000
        self.TOLVA_MAX = int(high)+100
        self.THRESHOLD_MAX = int(high) + 400 #3500

        mask_contour = cv2.bitwise_not(mask_contour)
        mask = cv2.bitwise_and(mask, mask, mask=mask_contour)

        mask = (mask // 255).astype(np.uint8)
        points = np.multiply(points, mask)
        
        return points, high

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Dist_to_Pixel(DIST,TMAX, TMIN):
        return (DIST-TMIN)/(TMAX-TMIN)
    
    @staticmethod
    @jit(nopython=True)
    def process_depth_image_numba(depth_image, DIST, TMAX, TMIN, color_init, color_end):
        H, W, _ = depth_image.shape
        for i in range(H):
            for j in range(W):
                if TMIN < DIST[i, j] < TMAX:
                    alpha = (DIST[i, j]-TMIN)/(TMAX-TMIN)
                    for k in range(3):
                        depth_image[i, j, k] = (1-alpha)*(color_end[k])+alpha*(color_init[k])

    def Put_colours(self, points):
        mask_over_max = points >= (self.THRESHOLD_MAX)
        mask_under_min = points < (self.THRESHOLD_MIN+self.TOLVA_MIN)/2

        self.depth_image[mask_over_max] = [0, 255, 0] # Green
        self.process_depth_image_numba(self.depth_image, points, self.THRESHOLD_MAX,self.TOLVA_MAX, self.COLOR_GREEN, self.COLOR_SYELLOW)
        self.process_depth_image_numba(self.depth_image, points, self.TOLVA_MAX,self.THRESHOLD_IDEAL, self.COLOR_SYELLOW, self.COLOR_YELLOW)
        self.process_depth_image_numba(self.depth_image, points, self.THRESHOLD_IDEAL,self.TOLVA_MIN, self.COLOR_YELLOW, self.COLOR_RED)
        self.process_depth_image_numba(self.depth_image, points, self.TOLVA_MIN,self.THRESHOLD_MIN, self.COLOR_RED, self.COLOR_BLACK)

    def Empty_spaces(self, points):
        mask_under_ideal = np.logical_and(points <= self.THRESHOLD_IDEAL - 500, points >= (self.TOLVA_MIN+self.THRESHOLD_MIN)/2) # (self.THRESHOLD_IDEAL+self.TOLVA_MIN)/2
        mask = np.zeros((points.shape[0], points.shape[1]), dtype=np.uint8)
        mask[mask_under_ideal] = 255

        (contornos,_) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contornos = [c for c in contornos if cv2.contourArea(c) > 2000]

        if len(contornos) > 0: 
            mask_spaces = cv2.drawContours(np.zeros((points.shape[0], points.shape[1]), dtype=np.uint8),contornos,-1,255, cv2.FILLED)
            point = np.multiply(points, (mask_spaces // 255).astype(np.uint8))
            self.process_depth_image_numba(self.depth_image, point, self.THRESHOLD_IDEAL-500, self.TOLVA_MIN, self.COLOR_SKYBLUE, self.COLOR_BLUE)
            self.process_depth_image_numba(self.depth_image, point, self.TOLVA_MIN, 1, self.COLOR_BLUE, self.COLOR_BLUE)
        return True if np.sum([cv2.contourArea(contorno) for contorno in contornos]) > self.UMBRAL_VACIO else False
   
    def analysis_side(self, mask_under_volcadura, mask_over_volcadura, vista, type, contornos):
        enviar_alerta = False
        h = mask_under_volcadura.shape[0]
        for c in contornos: 
            if cv2.contourArea(c) > 2000:
                x2, y2, w2, h2 = cv2.boundingRect(c) 
                mask_bulto = mask_over_volcadura[y2:y2+h2,x2:x2+w2]
                mask_bulto[np.isnan(mask_bulto)] = 0

                fila, columna = np.indices(mask_bulto.shape)
                centro_de_masa_x_out = int(np.round(np.average(columna, weights=mask_bulto),0))
                centro_de_masa_y_out = np.average(fila, weights=mask_bulto)
                valor_centro_de_masa_out = np.mean(mask_bulto[mask_bulto > self.TOLVA_MIN]) 

                mask_bulto = mask_under_volcadura[int(h/4):int(3*h/4),x2:x2+w2]
                mask_bulto[np.isnan(mask_bulto)] = 0

                fila, columna = np.indices(mask_bulto.shape)
                centro_de_masa_y_in = np.average(fila, weights=mask_bulto)
                valor_centro_de_masa_in = np.mean(mask_bulto[mask_bulto > self.TOLVA_MIN])

                if type:
                    indice_volcadura = (valor_centro_de_masa_out*(int(h/4)-(centro_de_masa_y_out+y2))-valor_centro_de_masa_in*(centro_de_masa_y_in))/1000
                else:
                    indice_volcadura = (valor_centro_de_masa_out*(centro_de_masa_y_out+y2)-valor_centro_de_masa_in*(int(2*h/4)-centro_de_masa_y_in))/1000
                #print('indice_volcadura',indice_volcadura)
                
                if indice_volcadura > self.UMBRAL_VOLCADURA: 
                    enviar_alerta = True
                    cv2.circle(vista, (x2+int(centro_de_masa_x_out ),y2+int(centro_de_masa_y_out)), 10, (255, 255, 255), -1)
                    print("Volcadura:", np.round(indice_volcadura,1))

        return enviar_alerta

    def Rollover(self, points, contour):
        (x, y, w, h) = contour # x, y, w, h = cv2.boundingRect(contour) 
        vista = self.depth_image.copy()

        mask_over_tolva_min   = np.zeros((points.shape[0], points.shape[1]))
        mask_upper_min  = (points >= (self.TOLVA_MIN))
        mask_over_tolva_min[mask_upper_min] = points[mask_upper_min]
        # mask_over_tolva_min = np.where(mask_over_tolva_min == 0, CAMERA_DIST-TOLVA_MIN, mask_over_tolva_min)

        mask_over_tolva_max   = np.zeros((points.shape[0], points.shape[1]))
        mask_upper_max  = (points >= (self.THRESHOLD_IDEAL)) #TOLVA_MAX
        mask_over_tolva_max[mask_upper_max] = points[mask_upper_max]

        mask_gray = np.zeros((points.shape[0], points.shape[1]), dtype=np.uint8)
        mask_gray[mask_upper_max] = 255

        (contornos,_) = cv2.findContours(mask_gray[y:y+int(h/4),x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_alert = self.analysis_side(mask_over_tolva_min[y:y+h,x:x+w], mask_over_tolva_max[y:y+int(h/4),x:x+w], vista[y:y+int(h/4),x:x+w], True, contornos)

        (contornos,_) = cv2.findContours(mask_gray[y+int(3*h/4):y+h,x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_alert = self.analysis_side(mask_over_tolva_min[y:y+h,x:x+w], mask_over_tolva_max[y+int(3*h/4):y+h,x:x+w], vista[y+int(3*h/4):y+h,x:x+w], False, contornos)

        return np.logical_or(right_alert, left_alert), mask_upper_min

    def Over_heigh(self, points):
        mask_height = points > self.THRESHOLD_MAX 
        mask_over_height = points > 4000
        
        return True if np.any(mask_height) else False, True if np.any(mask_over_height) else False

    def Volume(self, mask_over_tolva_min, depth_map_without_bordes):
        point3d = self.point3d.copy()
        new_point3d = np.zeros((self.__H, self.__W, 3))
        new_point3d[mask_over_tolva_min] = point3d[mask_over_tolva_min]

        length = (np.max(new_point3d[:,:,0]) - np.min(new_point3d[:,:,0]))/1000
        width = (np.max(new_point3d[:,:,1]) - np.min(new_point3d[:,:,1]))/1000

        ANCHOR = min(length,self.LENGTH)*min(width,self.WIDTH)
        VOLUMEN = np.round((np.nanmean(depth_map_without_bordes[depth_map_without_bordes != 0])/1000 - self.HEIGHT)*ANCHOR,1) # VOLUMEN = np.round(0.75*((CAMERA_DIST - TOLVA_MIN)/1000*2.4*9.5 - np.sum(np.power(mask_over_tolva_min/1000, 3)/3*10**-6) - 0.096),3) #np.count_nonzero(mask_under_volcadura>0)
        
        return VOLUMEN
    
    def Text(self, empty_alert, rollover_alert, height_alert, volume):
        
        def check(text, init_point, end_point):
            cv2.rectangle(self.depth_image, (init_point[0]+5,init_point[1]+5), (end_point[0]-5,end_point[1]-5), self.COLOR_WHITE,3)
            cv2.putText(self.depth_image, f'{text}', (end_point[0]+10,end_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX , 1.5, self.COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.line(self.depth_image, (init_point[0]+10,int((init_point[1]+end_point[1])/2)), (int((init_point[0]+end_point[0])/2),end_point[1]-10), self.COLOR_GREEN, 5)
            cv2.line(self.depth_image, (int((init_point[0]+end_point[0])/2),end_point[1]-10), (end_point[0]-10,init_point[1]+10), self.COLOR_GREEN, 5)   

        def cross(text, init_point, end_point):
            cv2.rectangle(self.depth_image, (init_point[0]+5,init_point[1]+5), (end_point[0]-5,end_point[1]-5), self.COLOR_WHITE,3)
            cv2.putText(self.depth_image, f'{text}', (end_point[0]+10,end_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX , 1.5, self.COLOR_RED, 2, cv2.LINE_AA)
            cv2.line(self.depth_image, (init_point[0]+10,init_point[1]+10), (end_point[0]-10,end_point[1]-10), self.COLOR_RED, 5)
            cv2.line(self.depth_image, (end_point[0]-10,init_point[1]+10), (init_point[0]+10,end_point[1]-10), self.COLOR_RED, 5)   

        check('Exceso de Vacios', (40, self.__H-190), (80, self.__H-150)) if not empty_alert else cross('Exceso de Vacios', (40, self.__H-190), (80, self.__H-150))
        check('Posibilidad de Volcadura', (40, self.__H-140), (80, self.__H-100)) if not rollover_alert else cross('Posibilidad de Volcadura', (40, self.__H-140), (80, self.__H-100))
        check('Exceso de Altura', (40, self.__H-90), (80, self.__H-50)) if not height_alert else cross('Exceso de Altura', (40, self.__H-90), (80, self.__H-50))

        cv2.putText(self.depth_image, f'VOLUMEN APROXIMADO: {volume} m3', (40, 80), cv2.FONT_HERSHEY_SIMPLEX , 2, self.COLOR_SKYBLUE, 3, cv2.LINE_AA) 
        
        cv2.putText(self.depth_image, f'Fecha: {self.date}', (self.__W-540,self.__H-100), cv2.FONT_HERSHEY_SIMPLEX , 1.5, self.COLOR_WHITE, 2, cv2.LINE_AA) 
        cv2.putText(self.depth_image, f' Hora: {self.hour}', (self.__W-540,self.__H-50), cv2.FONT_HERSHEY_SIMPLEX , 1.5, self.COLOR_WHITE, 2, cv2.LINE_AA) 

    def process(self):
        now = time.time()
        self.cam.retrieve_image(self.__object_image, sl.VIEW.LEFT)
        self.cam.retrieve_measure(self.__mat_cloud, sl.MEASURE.XYZRGBA)
        
        self.depth_image = np.zeros((self.__H,self.__W,3), dtype=np.uint8)
        self.image      = self.__object_image.get_data()[:, :, :3].copy()
        self.point3d    = self.Transformation(self.__mat_cloud.get_data()[:, :, :3][self.__Y:self.__YF, self.__X:self.__XF].copy())#self.Transformation(self.__mat_cloud.get_data()[:, :, :3][self.__Y:self.__YF, self.__X:self.__XF].copy()) #0.10
        depth           = self.point3d[:,:,2]
        mask, coords    = self.Segmentation(depth)
        
        if np.any(mask):
            self.__state[0, :-1] =  self.__state[0, 1:]
            self.__state[0, -1]  = True if np.abs(coords[0] + coords[2] / 2 - self.__center[0, -1]) <= 2 else False

            self.Put_colours(depth)
            depth_map_without_bordes, _ = self.Remove_contour(depth, mask)

            empty_alert = self.Empty_spaces(depth_map_without_bordes)                               # CONTORNOS DE VACÍOS           
            rollover_alert, mask_over_tolva_min = self.Rollover(depth_map_without_bordes, coords)   # ÍNDICE DE VOLCADURA
            height_alert, _ = self.Over_heigh(depth_map_without_bordes)                             # EXCESO DE ALTURA
            volume = self.Volume(mask_over_tolva_min, depth_map_without_bordes)                     # ESTIMACION DE VOLUMEN
            self.date, self.hour = self.UNIXtoDATE(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()) # FECHA
            
            self.Text(empty_alert, rollover_alert, height_alert, volume)
        else:
            self.Put_colours(depth)

            self.__state[0, :-1] =  self.__state[0, 1:]
            self.__state[0, -1]  = False
            self.print           = False

        self.__center[0, :-1] =  self.__center[0, 1:]
        self.__center[0, -1]  = coords[0] + coords[2] / 2

        # determinar si esta en la posición deseada y parado
        # if self.__state.all() == True and np.abs(np.mean(self.__center[0]) - self.__W/2) < self.POS_MAX and not self.print:
            # self.cant += 1
            # self.print = True
            # print('-------------save_results-----------------')
            # cv2.imwrite(f'{self.name} {self.cant}.png',cv2.resize(self.depth_image, (1920, 1080), interpolation = cv2.INTER_CUBIC))
            
        # print('self.__state:',self.__state)
        
        return self.image, self.depth_image
        
    def show3d(self):
        # points = cv2.GaussianBlur(points, (11, 11), 0)

        fig= go.Figure()

        x2 = self.point3d[:, :, 0]
        y2 = self.point3d[:, :, 1]
        self.point3d[:,:,2][self.mask_under_ideal] = self.TOLVA_MAX
        z2 = cv2.GaussianBlur(np.maximum(self.point3d[:, :, 2],0), (11, 11), 0) #points[:, :, 2] #points[:, :, 2]

        fig.add_surface(x=x2,
                        y=y2,
                        z=z2, 
                        colorscale=[[0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'],[0.8,'rgb(255,255,0)'],[1,'rgb(0,255,0)']],#'Viridis', 
                        opacity=1) 
        fig.update_layout(title='Me', 
                        autosize = False,  
                        scene_aspectratio=dict(x=2, y=1, z=1),
                          scene = dict(
                            xaxis = dict(visible=False),
                            yaxis = dict(visible=False),
                            zaxis =dict(visible=False)
                            ),
                        width = 1280,
                        height = 720,
                        showlegend=False,
                        )
        
        fig.update_traces(connectgaps=True)
        
        fig.write_html('filename.html')
        fig.show()

    @property
    def error(self):
        return True if self.cam.grab(self.__runtime) == sl.ERROR_CODE.SUCCESS else False

