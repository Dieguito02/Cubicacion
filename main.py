from cubage import Cubage
from drawing import vconcat_resize
from sendImage  import sendImg

import cv2

if __name__ == "__main__":
    rotation    = (0.05, 0, 0)
    route       = "./input/videos_SVO/HD2K_SN31048770_18-39-47.svo2"
    coordinates = (0,0,2208,1242)

    cubicacion = Cubage(rotation = rotation, cut_image = coordinates, name = "camion_noche")
    sendingImg = sendImg()
    # name = f'{route.split("/")[-1].split(".")[0]}.mp4'
    name = "4.mp4"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, 15, (960, 1080))
        
    while cubicacion.status:
        if cubicacion.error:
            img_original,  img_depth = cubicacion.process()
            draw_image = vconcat_resize([cv2.resize(img_original, (960,540), interpolation = cv2.INTER_CUBIC),cv2.resize(img_depth, (960,540), interpolation = cv2.INTER_CUBIC)])
            
            sendingImg.sendImage(draw_image.copy())   
            cv2.imshow('draw_img', draw_image)  
            out.write(draw_image)   

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cubicacion.show3d()
                break
        else:
            print("Error during capture")
            break

    # out.release()