const path = require('path');
const express = require('express');
const app = express();

const port = process.env.PORT || 3000;

console.log(`${__dirname}`);
app.use('/', express.static(`${__dirname}`));

// express will serve up index.html if it doesn't recognize the route
app.get('*', (req, res) => {
  res.sendFile(path.resolve(__dirname, 'public', 'index.html'));
});

app.listen(port, () => console.log(`Listening on port ${port}`));
