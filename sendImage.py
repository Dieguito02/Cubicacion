import threading
import cv2
import base64
from websocket_server import WebsocketServer

class sendImg:
    def __init__(self):
        self.server = WebsocketServer(port=8080, host='0.0.0.0')
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_client_left(self.client_left)
        self.server.set_fn_message_received(self.message_received)
        self.clients = []
        self.running = True  # Bandera para el hilo del servidor

        # Iniciar el servidor en un hilo separado
        self.server_thread = threading.Thread(target=self.server.run_forever)
        self.server_thread.start()

    def new_client(self, client, server):
        print(f"Cliente conectado: {client['id']}")
        self.clients.append(client)

    def client_left(self, client, server):
        print(f"Cliente desconectado: {client['id']}")
        self.clients.remove(client)

    def message_received(self, client, server, message):
        print(f"Mensaje recibido del cliente {client['id']}: {message}")

    def sendImage(self, img):
        # Codificar la imagen a base64
        _, buffer = cv2.imencode('.png', img)
        draw_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Enviar la imagen solo a los clientes conectados
        for client in self.clients:
            try:
                self.server.send_message(client, draw_image_b64)
            except Exception as e:
                print(f"Error al enviar la imagen al cliente {client['id']}: {e}")


    def stop(self):
        print("Deteniendo el servidor...")
        self.running = False
        self.server.server_close()  # Cerrar el servidor
        self.server_thread.join()  # Esperar a que el hilo del servidor termine

# class sendImg:
#     def __init__(self,port=8080, host='0.0.0.0'):
#         self.server = WebsocketServer(port=port, host=host)
#         self.server.set_fn_new_client(self.new_client)
#         self.server.set_fn_client_left(self.client_left)
#         self.server.set_fn_message_received(self.message_received)
#         self.clients = []  # Lista para almacenar clientes conectados

#         # Iniciar el servidor en un hilo separado
#         threading.Thread(target=self.server.run_forever).start()

#     def new_client(self, client, server):
#         print(f"Cliente conectado: {client['id']}")
#         self.clients.append(client)

#     def client_left(self, client, server):
#         print(f"Cliente desconectado: {client['id']}")
#         self.clients.remove(client)

#     def message_received(self, client, server, message):
#         print(f"Mensaje recibido del cliente {client['id']}: {message}")

