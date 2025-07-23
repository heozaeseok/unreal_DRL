import socket
import json

class SocketServer:
    def __init__(self, host='127.0.0.1', port=12345, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffer = ""
        self.last_action = None  # 마지막으로 보낸 액션 저장

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("[SERVER] Waiting for client...")
        self.client_socket, _ = self.server_socket.accept()
        self.client_socket.settimeout(self.timeout)
        print("[SERVER] Client connected.")

    def send(self, action_dict):
        """행동 데이터를 클라이언트로 전송"""
        message = json.dumps(action_dict) + "@"
        self.last_action = action_dict  # 마지막 전송 액션 저장
        self.client_socket.sendall(message.encode("utf-8"))

    def receive(self):
        """클라이언트로부터 JSON 또는 EpiDone 메시지를 수신"""
        try:
            while True:
                data = self.client_socket.recv(8192)
                if not data:
                    continue

                decoded = data.decode("utf-8", errors="ignore").strip()

                #단독 메시지 "EpiDone" 먼저 검사
                if decoded == "EpiDone":
                    print("[INFO] 'EpiDone' received directly from client.")
                    return "EpiDone"

                #버퍼에 누적 후 JSON 메시지 처리
                self.buffer += decoded
                while "@" in self.buffer:
                    raw_data, self.buffer = self.buffer.split("@", 1)
                    raw_data = raw_data.strip()

                    if not raw_data:
                        continue

                    try:
                        parsed = json.loads(raw_data)
                        return parsed
                    except json.JSONDecodeError:
                        print("[ERROR] Failed to parse JSON (skipping):", raw_data[:100])  # max 100 chars
                        continue

        except socket.timeout:
            print("[DEBUG] receive() timed out.")
            if self.last_action is not None:
                print("[DEBUG] Re-sending last action due to timeout.")
                try:
                    self.send(self.last_action)
                except Exception as e:
                    print(f"[ERROR] Re-send failed: {e}")
            return None

        except Exception as e:
            print(f"[ERROR] receive() exception: {e}")
            return None
            
    def close(self):
        """소켓 연결 종료"""
        try:
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            if hasattr(self, 'server_socket'):
                self.server_socket.close()
        except Exception as e:
            print(f"[ERROR] Socket close failed: {e}")
