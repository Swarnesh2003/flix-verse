import React, { useState } from 'react';
import axios from 'axios';
import {
  Button,
  TextField,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  AppBar,
  Toolbar,
  IconButton,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Container,
  Box,
  Paper,
  CircularProgress,
  Chip,
  ThemeProvider,
  createTheme,
  CssBaseline,
  alpha,
  Avatar
} from '@mui/material';
import { 
  Settings as SettingsIcon, 
  AccountCircle, 
  Logout, 
  MusicNote, 
  Sports, 
  Movie, 
  Theaters, 
  Star, 
  StarBorder,
  PlayArrow
} from '@mui/icons-material';

// Create a dark theme with blue accents
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3f80ff', // Bright blue
    },
    secondary: {
      main: '#00c2ff', // Cyan blue
    },
    background: {
      default: '#0c1929', // Very dark blue
      paper: '#152238',   // Slightly lighter dark blue
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0bec5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          transition: 'transform 0.2s, box-shadow 0.2s',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 10px 20px rgba(0, 0, 0, 0.3)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 3px 10px rgba(0, 0, 0, 0.2)',
          backgroundImage: 'linear-gradient(90deg, #152238 0%, #1e3a6d 100%)',
        },
      },
    },
  },
});

// API service using Axios
const apiService = {
  // Base URL - change to your Flask server address
  baseURL: 'http://localhost:5000/api',
  
  // Login method
  login: async (username, password) => {
    try {
      const response = await axios.post(`${apiService.baseURL}/login`, {
        username,
        password
      });
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },
  
  // Get video path method
  getVideoPath: async (videoId, preferences) => {
    try {
      const response = await axios.post(`${apiService.baseURL}/video`, {
        videoId,
        preferences
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching video:', error);
      throw error;
    }
  },
  
  // Get all videos
  getVideos: async () => {
    try {
      const response = await axios.get(`${apiService.baseURL}/videos`);
      return response.data;
    } catch (error) {
      console.error('Error fetching videos:', error);
      throw error;
    }
  }
};

const OTTPlatform = () => {
  // State variables
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [openPreferences, setOpenPreferences] = useState(false);
  const [preferences, setPreferences] = useState({
    skipSongs: false,
    skipFights: false,
    haveFights: false,
    cc:false
  });
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [videos, setVideos] = useState([]);
  const [category, setCategory] = useState('all');

  // Login handler
  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const response = await apiService.login(username, password);
      setIsLoggedIn(true);
      
      // Fetch videos after login
      try {
        const videosData = await apiService.getVideos();
        if (videosData && videosData.videos) {
          setVideos(videosData.videos);
        }
      } catch (videoError) {
        console.error('Error fetching videos:', videoError);
        setError('Could not load videos. Please try again later.');
      }
      
    } catch (error) {
      setError('Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  // Logout handler
  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
    setCurrentVideo(null);
  };

  // Preference handlers
  const handlePreferenceChange = (event) => {
    setPreferences({
      ...preferences,
      [event.target.name]: event.target.checked
    });
  };

  // Video selection handler
  const handleVideoSelect = async (videoId) => {
    setLoading(true);
    setSelectedVideo(videoId);
    setError('');
    
    try {
      const response = await apiService.getVideoPath(videoId, preferences);
      // Create the full URL for the video (including the server base URL)
      response.fullPath = `${apiService.baseURL.split('/api')[0]}${response.path}`;
      setCurrentVideo(response);
    } catch (error) {
      setError("Error fetching video. Please try again.");
      setSelectedVideo(null);
    } finally {
      setLoading(false);
    }
  };

  // Close video player
  const handleCloseVideo = () => {
    setCurrentVideo(null);
    setSelectedVideo(null);
  };

  // Filter videos by category
  const filteredVideos = category === 'all' 
    ? videos 
    : videos.filter(video => video.genre === category);

  // Get avatar letters from username
  const getAvatarLetters = (name) => {
    if (!name) return 'U';
    return name.charAt(0).toUpperCase();
  };

  // Get genres for filter buttons
  const genres = [...new Set(videos.map(video => video.genre))];

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      {!isLoggedIn ? (
        // Login screen
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundImage: 'linear-gradient(135deg, #0c1929 0%, #1e3a6d 100%)',
          }}
        >
          <Container component="main" maxWidth="xs">
            <Paper 
              elevation={6} 
              sx={{ 
                p: 4, 
                width: '100%', 
                borderRadius: 3,
                backgroundImage: 'linear-gradient(135deg, #152238 0%, #1a2c4d 100%)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
              }}
            >
              <Box 
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <Box 
                  sx={{ 
                    mb: 3, 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 80, 
                    height: 80, 
                    borderRadius: '50%',
                    background: 'linear-gradient(45deg, #3f80ff, #00c2ff)',
                    boxShadow: '0 5px 15px rgba(0,0,0,0.3)'
                  }}
                >
                  <Theaters sx={{ fontSize: 40, color: 'white' }} />
                </Box>
                <Typography component="h1" variant="h4" sx={{ mb: 3, color: 'white', fontWeight: 700 }}>
                  FlixVerse
                </Typography>
                <form onSubmit={handleLogin} style={{ width: '100%' }}>
                  <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    id="username"
                    label="Username"
                    name="username"
                    autoComplete="username"
                    autoFocus
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 2,
                        backgroundColor: alpha('#ffffff', 0.05),
                      }
                    }}
                  />
                  <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    name="password"
                    label="Password"
                    type="password"
                    id="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 2,
                        backgroundColor: alpha('#ffffff', 0.05),
                      }
                    }}
                  />
                  {error && (
                    <Typography color="error" variant="body2" sx={{ mt: 1 }}>
                      {error}
                    </Typography>
                  )}
                  <Button
                    type="submit"
                    fullWidth
                    variant="contained"
                    color="primary"
                    sx={{ 
                      mt: 3, 
                      mb: 2, 
                      py: 1.2,
                      fontSize: '1rem',
                      background: 'linear-gradient(45deg, #3f80ff, #00c2ff)',
                      boxShadow: '0 5px 15px rgba(0,0,0,0.3)',
                      '&:hover': {
                        background: 'linear-gradient(45deg, #3373e8, #00b3ec)',
                      }
                    }}
                    disabled={loading}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Sign In'}
                  </Button>
                </form>
              </Box>
            </Paper>
          </Container>
        </Box>
      ) : (
        // Main app UI after login
        <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          {/* App Bar */}
          <AppBar position="fixed">
            <Toolbar>
              <Box display="flex" alignItems="center">
                <Theaters sx={{ mr: 1 }} />
                <Typography variant="h6" component="div" sx={{ fontWeight: 700 }}>
                  FlixVerse
                </Typography>
              </Box>
              
              <Box sx={{ flexGrow: 1 }} />
              
              <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                <Chip
                  icon={<MusicNote />}
                  label={preferences.skipSongs ? "Songs: Skip" : "Songs: Show"}
                  color={preferences.skipSongs ? "secondary" : "default"}
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Chip
                  icon={<Sports />}
                  label={preferences.skipFights ? "Fights: Skip" : "Fights: Show"}
                  color={preferences.skipFights ? "secondary" : "default"}
                  size="small"
                />
              </Box>
              <IconButton
                color="inherit"
                onClick={() => setOpenPreferences(true)}
                sx={{ 
                  mr: 1,
                  '&:hover': {
                    background: alpha('#ffffff', 0.1)
                  }
                }}
              >
                <SettingsIcon />
              </IconButton>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Avatar 
                  sx={{ 
                    bgcolor: 'primary.main', 
                    width: 36, 
                    height: 36,
                    mr: 1,
                    boxShadow: '0 0 10px rgba(63, 128, 255, 0.5)'
                  }}
                >
                  {getAvatarLetters(username)}
                </Avatar>
                <Typography variant="body1" sx={{ display: { xs: 'none', sm: 'block' } }}>
                  {username}
                </Typography>
                <IconButton 
                  color="inherit" 
                  onClick={handleLogout} 
                  sx={{ 
                    ml: 1,
                    '&:hover': {
                      background: alpha('#ffffff', 0.1)
                    }
                  }}
                >
                  <Logout />
                </IconButton>
              </Box>
            </Toolbar>
          </AppBar>
          <Toolbar /> {/* Spacer for fixed AppBar */}

          {/* Main Content */}
          <Container sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
            <Box sx={{ mb: 4 }}>
              <Typography variant="h4" gutterBottom sx={{ 
                fontWeight: 700, 
                color: 'white',
                display: 'flex',
                alignItems: 'center' 
              }}>
                <Star sx={{ mr: 1, color: '#FFD700' }} />
                Discover Movies
              </Typography>
              
              {/* Category filter buttons */}
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2, mb: 3 }}>
                <Button 
                  variant={category === 'all' ? 'contained' : 'outlined'}
                  onClick={() => setCategory('all')}
                  sx={{ borderRadius: 5 }}
                >
                  All
                </Button>
                {genres.map(genre => (
                  <Button
                    key={genre}
                    variant={category === genre ? 'contained' : 'outlined'}
                    onClick={() => setCategory(genre)}
                    sx={{ borderRadius: 5, textTransform: 'capitalize' }}
                  >
                    {genre}
                  </Button>
                ))}
              </Box>
            </Box>
            
            {error && (
              <Paper 
                elevation={4} 
                sx={{ 
                  p: 2, 
                  mb: 4, 
                  borderLeft: '4px solid #f44336',
                  backgroundColor: alpha('#f44336', 0.1)
                }}
              >
                <Typography color="error" variant="body1">
                  {error}
                </Typography>
              </Paper>
            )}
            
            {videos.length === 0 ? (
              <Box 
                sx={{ 
                  mt: 8, 
                  textAlign: 'center',
                  p: 4,
                  backgroundColor: alpha('#ffffff', 0.05),
                  borderRadius: 3
                }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <CircularProgress color="secondary" size={60} thickness={4} />
                    <Typography variant="h6" sx={{ mt: 2 }}>
                      Loading videos...
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="h6">
                    No videos available
                  </Typography>
                )}
              </Box>
            ) : (
              filteredVideos.length === 0 ? (
                <Typography variant="h6" sx={{ textAlign: 'center', mt: 4 }}>
                  No videos found in this category
                </Typography>
              ) : (
                <Grid container spacing={3}>
                  {filteredVideos.map((video) => (
                    <Grid item xs={12} sm={6} md={4} key={video.id}>
                      <Card 
                        sx={{ 
                          cursor: 'pointer', 
                          height: '100%', 
                          position: 'relative',
                          overflow: 'hidden',
                          backgroundColor: '#1a2c4d',
                        }}
                        onClick={() => handleVideoSelect(video.id)}
                      >
                        <Box sx={{ position: 'relative' }}>
                          <CardMedia
                            component="img"
                            height="180"
                            image="thumbnail.jpg"
                            alt={video.title}
                            sx={{ 
                              transition: 'transform 0.3s',
                              '&:hover': {
                                transform: 'scale(1.05)'
                              }
                            }}
                          />
                          <Box 
                            sx={{ 
                              position: 'absolute', 
                              top: 0, 
                              left: 0, 
                              width: '100%', 
                              height: '100%',
                              background: 'linear-gradient(to top, rgba(21, 34, 56, 0.9) 0%, rgba(21, 34, 56, 0) 50%)',
                              transition: 'opacity 0.3s',
                              opacity: 0.7,
                              '&:hover': {
                                opacity: 0.3
                              }
                            }} 
                          />
                          <Box
                            sx={{
                              position: 'absolute',
                              top: '50%',
                              left: '50%',
                              transform: 'translate(-50%, -50%)',
                              width: 60,
                              height: 60,
                              borderRadius: '50%',
                              backgroundColor: alpha('#3f80ff', 0.9),
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              transition: 'opacity 0.3s, transform 0.3s',
                              opacity: 0,
                              boxShadow: '0 5px 15px rgba(0,0,0,0.5)',
                              '&:hover': {
                                transform: 'translate(-50%, -50%) scale(1.1)',
                              },
                              '.MuiCard-root:hover &': {
                                opacity: 1,
                              }
                            }}
                          >
                            <PlayArrow sx={{ fontSize: 30 }} />
                          </Box>
                        </Box>
                        <CardContent sx={{ position: 'relative', zIndex: 1 }}>
                          <Typography gutterBottom variant="h6" component="div" sx={{ fontWeight: 600 }}>
                            {video.title}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                color: 'primary.light', 
                                textTransform: 'uppercase',
                                fontWeight: 600,
                                fontSize: '0.7rem',
                                letterSpacing: 1,
                              }}
                            >
                              {video.genre}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <StarBorder sx={{ fontSize: 16, color: '#FFD700', mr: 0.5 }} />
                              <Typography variant="body2" color="text.secondary">
                                {Math.floor(Math.random() * 2) + 3}.{Math.floor(Math.random() * 10)}/5
                              </Typography>
                            </Box>
                          </Box>
                          <Box sx={{ mt: 1, display: 'flex', gap: 0.5 }}>
                            {video.hasSongs && (
                              <Chip 
                                icon={<MusicNote sx={{ fontSize: '0.8rem' }} />} 
                                label="Songs" 
                                size="small"
                                sx={{ 
                                  bgcolor: alpha('#3f80ff', 0.2),
                                  '& .MuiChip-label': {
                                    fontSize: '0.7rem',
                                  }
                                }}
                              />
                            )}
                            {video.hasFights && (
                              <Chip 
                                icon={<Sports sx={{ fontSize: '0.8rem' }} />} 
                                label="Fights" 
                                size="small"
                                sx={{ 
                                  bgcolor: alpha('#00c2ff', 0.2),
                                  '& .MuiChip-label': {
                                    fontSize: '0.7rem',
                                  }
                                }}
                              />
                            )}
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )
            )}
          </Container>

          {/* Preferences Dialog */}
          <Dialog 
            open={openPreferences} 
            onClose={() => setOpenPreferences(false)}
            PaperProps={{
              sx: {
                borderRadius: 3,
                backgroundImage: 'linear-gradient(135deg, #152238 0%, #1a2c4d 100%)',
              }
            }}
          >
            <DialogTitle sx={{ borderBottom: `1px solid ${alpha('#ffffff', 0.1)}` }}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>Content Preferences</Typography>
            </DialogTitle>
            <DialogContent sx={{ mt: 2 }}>
              <Typography variant="body1" gutterBottom>
                Select content to skip during playback:
              </Typography>
              <FormGroup>
                <FormControlLabel 
                  control={
                    <Checkbox 
                      checked={preferences.skipSongs} 
                      onChange={handlePreferenceChange} 
                      name="skipSongs" 
                      sx={{
                        color: alpha('#3f80ff', 0.7),
                        '&.Mui-checked': {
                          color: '#3f80ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <MusicNote sx={{ mr: 1, fontSize: 20, color: '#3f80ff' }} />
                      <Typography>Skip Songs</Typography>
                    </Box>
                  }
                />
                <FormControlLabel 
                  control={
                    <Checkbox 
                      checked={preferences.skipFights} 
                      onChange={handlePreferenceChange} 
                      name="skipFights" 
                      sx={{
                        color: alpha('#00c2ff', 0.7),
                        '&.Mui-checked': {
                          color: '#00c2ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Sports sx={{ mr: 1, fontSize: 20, color: '#00c2ff' }} />
                      <Typography>Skip Fight Scenes</Typography>
                    </Box>
                  }
                />
                <FormControlLabel 
                  control={
                    <Checkbox 
              
                      sx={{
                        color: alpha('#00c2ff', 0.7),
                        '&.Mui-checked': {
                          color: '#00c2ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Sports sx={{ mr: 1, fontSize: 20, color: '#00c2ff' }} />
                      <Typography>Skip Mature Content</Typography>
                    </Box>
                  }
                />
                <FormControlLabel 
                  control={
                    <Checkbox 
                    checked={preferences.haveFights} 
                    onChange={handlePreferenceChange} 
                    name="haveFights" 
                      sx={{
                        color: alpha('#00c2ff', 0.7),
                        '&.Mui-checked': {
                          color: '#00c2ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Sports sx={{ mr: 1, fontSize: 20, color: '#00c2ff' }} />
                      <Typography>Play only Fight Scenes</Typography>
                    </Box>
                  }
                />
                <FormControlLabel 
                  control={
                    <Checkbox 
                      
                      sx={{
                        color: alpha('#00c2ff', 0.7),
                        '&.Mui-checked': {
                          color: '#00c2ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Sports sx={{ mr: 1, fontSize: 20, color: '#00c2ff' }} />
                      <Typography>Play only songs</Typography>
                    </Box>
                  }
                />
                <FormControlLabel 
                  control={
                    <Checkbox 
                    checked={preferences.cc} 
                    onChange={handlePreferenceChange} 
                    name="cc"
                      sx={{
                        color: alpha('#00c2ff', 0.7),
                        '&.Mui-checked': {
                          color: '#00c2ff',
                        },
                      }}
                    />
                  } 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Sports sx={{ mr: 1, fontSize: 20, color: '#00c2ff' }} />
                      <Typography>Generate cc</Typography>
                    </Box>
                  }
                />
              </FormGroup>
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 3 }}>
              <Button 
                onClick={() => setOpenPreferences(false)}
                variant="contained"
                sx={{ 
                  background: 'linear-gradient(45deg, #3f80ff, #00c2ff)',
                  px: 3
                }}
              >
                Save Preferences
              </Button>
            </DialogActions>
          </Dialog>

          {/* Video Player Dialog */}
          <Dialog
            fullWidth
            maxWidth="md"
            open={currentVideo !== null}
            onClose={handleCloseVideo}
            PaperProps={{
              sx: {
                borderRadius: 3,
                backgroundImage: 'linear-gradient(135deg, #0c1929 0%, #152238 100%)',
                overflow: 'hidden'
              }
            }}
          >
            {currentVideo && (
              <>
                <DialogTitle sx={{ 
                  borderBottom: `1px solid ${alpha('#ffffff', 0.1)}`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {currentVideo.title}
                    </Typography>
                    <Typography variant="subtitle2" color="text.secondary">
                      Version: {currentVideo.version}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {currentVideo.preferences.skipSongs && (
                      <Chip 
                        size="small" 
                        color="primary" 
                        label="Songs Removed" 
                        icon={<MusicNote />} 
                      />
                    )}
                    {currentVideo.preferences.skipFights && (
                      <Chip 
                        size="small" 
                        color="secondary" 
                        label="Fights Removed" 
                        icon={<Sports />} 
                      />
                    )}
                  </Box>
                </DialogTitle>
                <DialogContent sx={{ p: 0 }}>
                  <Box sx={{ position: 'relative', pt: '56.25%' /* 16:9 aspect ratio */ }}>
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        bgcolor: '#000',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexDirection: 'column'
                      }}
                    >
                      {loading ? (
                        <CircularProgress 
                          color="secondary" 
                          size={60}
                          thickness={4}
                        />
                      ) : (
                        <video 
                          width="100%" 
                          height="100%" 
                          controls 
                          autoPlay
                          style={{ 
                            boxShadow: '0 0 20px rgba(0, 0, 0, 0.5)',
                            outline: 'none'
                          }}
                        >
                          <source src={`${apiService.baseURL.split('/api')[0]}${currentVideo.path}`} type="video/mp4" />
                          Your browser does not support the video tag.
                        </video>
                      )}
                    </Box>
                  </Box>
                </DialogContent>
                <DialogActions sx={{ px: 3, py: 2 }}>
                  <Button 
                    onClick={handleCloseVideo}
                    variant="outlined"
                    color="secondary"
                    sx={{ px: 3 }}
                  >
                    Close
                  </Button>
                </DialogActions>
              </>
            )}
          </Dialog>
        </Box>
      )}
    </ThemeProvider>
  );
};

export default OTTPlatform;