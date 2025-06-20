import socket
import json

class SocketServer:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.buffer = ""

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("[SERVER] Waiting for client...")
        self.client_socket, _ = self.server_socket.accept()
        print("[SERVER] Client connected.")

    def send(self, action_dict):
        message = json.dumps(action_dict) + "@"
        self.client_socket.sendall(message.encode("utf-8"))
        #print(f"[SEND]: {message.strip()}")

    def receive(self):
        while True:
            data = self.client_socket.recv(2048)
            if not data:
                continue

            self.buffer += data.decode("utf-8", errors="ignore")
            if "@" in self.buffer:
                #print(f"[RECEIVE BUFFER]: {self.buffer}")
                raw_data = self.buffer.split("@")[0].strip()
                self.buffer = self.buffer[len(raw_data)+1:]
                try:
                    parsed = json.loads(raw_data)
                    #print(f"[PARSED DATA]: {parsed}")
                    return parsed
                except json.JSONDecodeError as e:
                    #print(f"[JSON ERROR]: {e} from raw: {raw_data}")
                    return None

    def close(self):
        try:
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            if hasattr(self, 'server_socket'):
                self.server_socket.close()
        except Exception as e:
            print(f"[ERROR] Socket close failed: {e}")
