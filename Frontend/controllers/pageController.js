const apiBaseUrl = process.env.API_BASE_URL || 'http://127.0.0.1:8000';

exports.home = (req, res) => {
  res.render('index', {
    title: 'GenreAI — Discover Intelligence Behind Every Genre',
    activeNav: 'home',
    apiBaseUrl
  });
};

exports.dashboard = (req, res) => {
  res.render('dashboard', {
    title: 'Dashboard — GenreAI',
    activeNav: 'dashboard',
    apiBaseUrl
  });
};

exports.features = (req, res) => {
  res.render('features', {
    title: 'Features — GenreAI',
    activeNav: 'features',
    apiBaseUrl
  });
};
