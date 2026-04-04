const express = require('express');
const router = express.Router();
const pageController = require('../routes/pageController');

router.get('/', pageController.home);
router.get('/dashboard', pageController.dashboard);
router.get('/features', pageController.features);

module.exports = router;
