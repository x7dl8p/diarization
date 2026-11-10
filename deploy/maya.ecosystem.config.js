// pm2 app for the maya-transcribe ASR service (diarai/maya_server.py), used by diarai when a request asks for
// model=maya. localhost only (port 8086); diarai calls it, nothing else should.
//   pm2 start C:\Users\ali\diarai\deploy\maya.ecosystem.config.js     then: pm2 save
// Own virtualenv (C:\Users\ali\diarai\maya\.venv: Python 3.12, torch 2.14.1+cu126, transformers 5.20.2) because
// diarai's DiariZen pins torch 2.1. GPU: the RTX 2060 (device 0 in FASTEST_FIRST order), next to diarai.
module.exports = {
  apps: [
    {
      name: "diarai-maya",
      cwd: "C:\\Users\\ali\\diarai",
      script: "C:\\Users\\ali\\diarai\\maya\\.venv\\Scripts\\python.exe",
      args: "diarai\\maya_server.py",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 10000,
      env: {
        MAYA_MODEL: "C:\\Users\\ali\\diarai\\models\\maya-transcribe",
        MAYA_DEVICE: "cuda:0",
        MAYA_PORT: "8086",
        HF_HUB_OFFLINE: "1",
        CUDA_DEVICE_ORDER: "FASTEST_FIRST",
        CUDA_VISIBLE_DEVICES: "0",
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
    },
  ],
};
