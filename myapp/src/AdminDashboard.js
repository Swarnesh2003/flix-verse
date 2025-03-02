import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  CircularProgress,
  Alert,
  Snackbar,
  Tabs,
  Tab,
  Chip,
  Divider,
  Card,
  CardContent,
  IconButton
} from '@mui/material';
import { 
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  CloudUpload as CloudUploadIcon,
  Refresh as RefreshIcon,
  Movie as MovieIcon
} from '@mui/icons-material';

// Configure axios base URL
const api = axios.create({
  baseURL: 'http://localhost:5000'
});

const AdminDashboard = () => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [versionLoading, setVersionLoading] = useState(false);
  const [openUploadDialog, setOpenUploadDialog] = useState(false);
  const [openVersionDialog, setOpenVersionDialog] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // Form states
  const [uploadForm, setUploadForm] = useState({
    title: '',
    genre: 'action',
    hasSongs: true,
    hasFights: true,
    videoFile: null
  });
  
  const [versionForm, setVersionForm] = useState({
    videoId: '',
    versionType: 'fightRemoved',
    videoFile: null
  });

  const genres = ['action', 'comedy', 'drama', 'romance', 'scifi', 'thriller', 'horror', 'documentary', 'other'];
  const versionTypes = [
    { value: 'fightRemoved', label: 'Fights Removed' },
    { value: 'songsRemoved', label: 'Songs Removed' },
    { value: 'bothRemoved', label: 'Songs & Fights Removed' }
  ];

  // Load videos on component mount
  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/admin/videos');
      setVideos(response.data.videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
      showNotification('Failed to load videos', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const handleUploadDialogOpen = () => {
    setOpenUploadDialog(true);
  };

  const handleUploadDialogClose = () => {
    setOpenUploadDialog(false);
    resetUploadForm();
  };

  const handleVersionDialogOpen = (video) => {
    setSelectedVideo(video);
    setVersionForm({
      ...versionForm,
      videoId: video.id
    });
    setOpenVersionDialog(true);
  };

  const handleVersionDialogClose = () => {
    setOpenVersionDialog(false);
    setSelectedVideo(null);
    resetVersionForm();
  };

  const resetUploadForm = () => {
    setUploadForm({
      title: '',
      genre: 'action',
      hasSongs: true,
      hasFights: true,
      videoFile: null
    });
  };

  const resetVersionForm = () => {
    setVersionForm({
      videoId: '',
      versionType: 'fightRemoved',
      videoFile: null
    });
  };

  const handleUploadChange = (e) => {
    const { name, value, type, checked, files } = e.target;
    
    if (type === 'file') {
      setUploadForm({
        ...uploadForm,
        [name]: files[0]
      });
    } else if (type === 'checkbox') {
      setUploadForm({
        ...uploadForm,
        [name]: checked
      });
    } else {
      setUploadForm({
        ...uploadForm,
        [name]: value
      });
    }
  };

  const handleVersionChange = (e) => {
    const { name, value, files } = e.target;
    
    if (name === 'videoFile') {
      setVersionForm({
        ...versionForm,
        [name]: files[0]
      });
    } else {
      setVersionForm({
        ...versionForm,
        [name]: value
      });
    }
  };

  const showNotification = (message, severity = 'success') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    
    if (!uploadForm.videoFile) {
      showNotification('Please select a video file', 'error');
      return;
    }
    
    setUploadLoading(true);
    
    const formData = new FormData();
    formData.append('title', uploadForm.title);
    formData.append('genre', uploadForm.genre);
    formData.append('hasSongs', uploadForm.hasSongs);
    formData.append('hasFights', uploadForm.hasFights);
    formData.append('videoFile', uploadForm.videoFile);
    
    try {
      const response = await api.post('/api/admin/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      showNotification('Video uploaded successfully');
      handleUploadDialogClose();
      fetchVideos();
    } catch (error) {
      console.error('Error uploading video:', error);
      const errorMessage = error.response?.data?.error || 'Failed to upload video';
      showNotification(errorMessage, 'error');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleVersionSubmit = async (e) => {
    e.preventDefault();
    
    if (!versionForm.videoFile) {
      showNotification('Please select a video file', 'error');
      return;
    }
    
    setVersionLoading(true);
    
    const formData = new FormData();
    formData.append('videoId', versionForm.videoId);
    formData.append('versionType', versionForm.versionType);
    formData.append('videoFile', versionForm.videoFile);
    
    try {
      const response = await api.post('/api/admin/upload-version', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      showNotification('Video version uploaded successfully');
      handleVersionDialogClose();
      fetchVideos();
    } catch (error) {
      console.error('Error uploading video version:', error);
      const errorMessage = error.response?.data?.error || 'Failed to upload video version';
      showNotification(errorMessage, 'error');
    } finally {
      setVersionLoading(false);
    }
  };

  const getVersionLabel = (versionKey) => {
    switch(versionKey) {
      case 'original': return 'Original';
      case 'fightRemoved': return 'Fights Removed';
      case 'songsRemoved': return 'Songs Removed';
      case 'bothRemoved': return 'Songs & Fights Removed';
      default: return versionKey;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Admin Dashboard
          </Typography>
          <Box>
            <Button 
              variant="contained" 
              color="primary"
              startIcon={<RefreshIcon />}
              onClick={fetchVideos}
              sx={{ mr: 2 }}
            >
              Refresh
            </Button>
            <Button 
              variant="contained" 
              color="secondary"
              startIcon={<AddIcon />}
              onClick={handleUploadDialogOpen}
            >
              Upload New Video
            </Button>
          </Box>
        </Box>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange}>
            <Tab label="Video Management" />
            <Tab label="Analytics" />
            <Tab label="Settings" />
          </Tabs>
        </Box>
        
        {currentTab === 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Video Library
            </Typography>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : videos.length === 0 ? (
              <Alert severity="info">
                No videos available. Click "Upload New Video" to add content.
              </Alert>
            ) : (
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>ID</TableCell>
                      <TableCell>Title</TableCell>
                      <TableCell>Genre</TableCell>
      
                      <TableCell>Available Versions</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {videos.map((video) => (
                      <TableRow key={video.id}>
                        <TableCell>{video.id}</TableCell>
                        <TableCell>{video.title}</TableCell>
                        <TableCell>{video.genre}</TableCell>
                       
                        <TableCell>
                          {video.versions.map((version) => (
                            <Chip 
                              key={version} 
                              label={getVersionLabel(version)} 
                              variant="outlined" 
                              size="small" 
                              sx={{ mr: 1, mb: 1 }} 
                            />
                          ))}
                        </TableCell>
                        <TableCell>
                          <IconButton 
                            color="primary"
                            onClick={() => handleVersionDialogOpen(video)}
                            title="Upload version"
                          >
                            <CloudUploadIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Box>
        )}
        
        {currentTab === 1 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6">
              Analytics Dashboard
            </Typography>
            <Typography variant="body1" sx={{ mt: 2, color: 'text.secondary' }}>
              Analytics features will be implemented in future updates.
            </Typography>
          </Box>
        )}
        
        {currentTab === 2 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6">
              System Settings
            </Typography>
            <Typography variant="body1" sx={{ mt: 2, color: 'text.secondary' }}>
              Settings configuration will be implemented in future updates.
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Upload Video Dialog */}
      <Dialog open={openUploadDialog} onClose={handleUploadDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>Upload New Video</DialogTitle>
        <form onSubmit={handleUploadSubmit}>
          <DialogContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  name="title"
                  label="Video Title"
                  fullWidth
                  value={uploadForm.title}
                  onChange={handleUploadChange}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Genre</InputLabel>
                  <Select
                    name="genre"
                    value={uploadForm.genre}
                    onChange={handleUploadChange}
                    label="Genre"
                  >
                    {genres.map((genre) => (
                      <MenuItem key={genre} value={genre}>
                        {genre.charAt(0).toUpperCase() + genre.slice(1)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
       
              <Grid item xs={12}>
                <Box sx={{ border: '1px dashed grey', p: 3, textAlign: 'center' }}>
                  <input
                    accept="video/*"
                    id="video-upload"
                    type="file"
                    name="videoFile"
                    onChange={handleUploadChange}
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="video-upload">
                    <Button
                      variant="contained"
                      component="span"
                      startIcon={<CloudUploadIcon />}
                    >
                      Select Video File
                    </Button>
                  </label>
                  {uploadForm.videoFile && (
                    <Typography variant="body1" sx={{ mt: 2 }}>
                      Selected file: {uploadForm.videoFile.name}
                    </Typography>
                  )}
                </Box>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleUploadDialogClose}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary"
              disabled={uploadLoading || !uploadForm.title || !uploadForm.videoFile}
            >
              {uploadLoading ? <CircularProgress size={24} /> : 'Upload'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Upload Version Dialog */}
      <Dialog open={openVersionDialog} onClose={handleVersionDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>
          Upload Alternative Version
          {selectedVideo && (
            <Typography variant="subtitle1" color="text.secondary">
              For: {selectedVideo.title}
            </Typography>
          )}
        </DialogTitle>
        <form onSubmit={handleVersionSubmit}>
          <DialogContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Version Type</InputLabel>
                  <Select
                    name="versionType"
                    value={versionForm.versionType}
                    onChange={handleVersionChange}
                    label="Version Type"
                  >
                    {versionTypes.map((type) => (
                      <MenuItem key={type.value} value={type.value}>
                        {type.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ border: '1px dashed grey', p: 3, textAlign: 'center' }}>
                  <input
                    accept="video/*"
                    id="version-upload"
                    type="file"
                    name="videoFile"
                    onChange={handleVersionChange}
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="version-upload">
                    <Button
                      variant="contained"
                      component="span"
                      startIcon={<CloudUploadIcon />}
                    >
                      Select Version File
                    </Button>
                  </label>
                  {versionForm.videoFile && (
                    <Typography variant="body1" sx={{ mt: 2 }}>
                      Selected file: {versionForm.videoFile.name}
                    </Typography>
                  )}
                </Box>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleVersionDialogClose}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary"
              disabled={versionLoading || !versionForm.videoFile}
            >
              {versionLoading ? <CircularProgress size={24} /> : 'Upload Version'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default AdminDashboard;