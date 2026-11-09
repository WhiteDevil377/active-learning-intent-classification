"""Local batch intent-prediction API."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from predict import Predictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8766)
    args=parser.parse_args()
    model=Predictor()

    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,body):
            data=json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path=="/health":
                self.respond(200,{"status":"ok","intents":len(model.bundle["labels"]),
                                  "training_labels":model.bundle["selection"]["labels"]})
            else:
                self.respond(404,{"error":"Use /health or POST /predict"})

        def do_POST(self):
            if self.path!="/predict":
                self.respond(404,{"error":"Use POST /predict"})
                return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=65536:
                    raise ValueError("Invalid request size")
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):
                    raise ValueError("Expected a JSON object containing texts")
                output=model.predict(body.get("texts"))
                self.respond(200,{"predictions":output})
            except (ValueError,TypeError) as exc:
                self.respond(400,{"error":str(exc)})

    server=HTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Local API: http://127.0.0.1:{args.port}",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
