# Phone Demo Roadmap — Phase 5.2.28

## Fastest client demo: same Wi-Fi

1. Put PC and phone on the same Wi-Fi.
2. Backend:
   `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
3. Frontend:
   `python -m http.server 5500 --bind 0.0.0.0`
4. Run `ipconfig` on Windows and find the PC IPv4 address.
5. On the phone open `http://<PC-IP>:5500`.
6. If Windows Firewall blocks it, allow TCP 5500 and 8000 on the private network.

The PC remains the GPU server. The phone is only the browser client.

## Remote demo

Use a tunnel/reverse proxy rather than exposing Uvicorn directly to the Internet. Recommended architecture:

Internet/phone → HTTPS demo URL → reverse proxy/tunnel → frontend + `/api/*` → FastAPI on the GPU PC.

Before doing this, make the frontend API base URL configurable or put frontend and backend behind the same HTTPS origin. Add an access-control layer before public exposure.

## Production later

Move GPU inference to a dedicated CUDA server, add authentication, request limits, job/status handling, persistent storage, monitoring and a proper deployment pipeline.
