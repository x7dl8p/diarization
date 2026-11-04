// pm2 app for the diarai speech-to-text + diarization API (https://sarvam.androcoders.net) on port 8085 (open in the firewall as ocr-http-8085; OCR and diarai cannot run together).
//   pm2 start C:\Users\ali\diarai\diarai.ecosystem.config.js     then: pm2 save
// GPU: the RTX 2060 only (CUDA device 0 in FASTEST_FIRST order). The GTX 1060 stays free.
module.exports = {
  apps: [
    {
      name: "diarai",
      cwd: "C:\\Users\\ali\\diarai",
      script: "C:\\Users\\ali\\diarai\\.venv\\Scripts\\python.exe",
      args: "-m diarai serve --host 0.0.0.0 --port 8085 --log-level warning",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 15000,
      env: {
        HF_HUB_OFFLINE: "1",
        HF_HUB_DISABLE_XET: "1",
        CUDA_DEVICE_ORDER: "FASTEST_FIRST",
        CUDA_VISIBLE_DEVICES: "0",
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
    },
  ],
};
