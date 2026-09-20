require('dotenv').config();
const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// View engine setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static files
app.use(express.static(path.join(__dirname, 'public')));

// Body parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
const pageRoutes = require('./routes/index');
app.use('/', pageRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).render('index', {
    title: '404 — GenreAI',
    activeNav: '',
    apiBaseUrl: process.env.API_BASE_URL || 'http://127.0.0.1:8000'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n  ⚡ GenreAI server running at http://localhost:${PORT}\n`);
});
