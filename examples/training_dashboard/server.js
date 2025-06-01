const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');

const index = fs.readFileSync(__dirname + '/dashboard.html');
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(index);
});

const wss = new WebSocket.Server({ server });
let training = null;

wss.on('connection', ws => {
  if (!training) {
    training = ws;
    ws.on('message', msg => {
      for (const client of wss.clients) {
        if (client !== training && client.readyState === WebSocket.OPEN) {
          client.send(msg);
        }
      }
    });
    ws.on('close', () => { training = null; });
  }
});

const port = 8080;
server.listen(port, () => {
  console.log(`Dashboard running on http://localhost:${port}`);
});
