import React, { useState, useEffect } from 'react';
import OTTPlatform from './OTTPlatform';
import AdminDashboard from './AdminDashboard';
import { createTheme, ThemeProvider, CssBaseline, CircularProgress, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './LoginPage'; // We'll create this component next

// Create a theme instance
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
  },
});

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Check if user is already logged in on component mount
  useEffect(() => {
    const checkAuthStatus = () => {
      const storedAuth = localStorage.getItem('isAuthenticated');
      const storedRole = localStorage.getItem('userRole');
      
      if (storedAuth === 'true' && storedRole) {
        setIsAuthenticated(true);
        setUserRole(storedRole);
      }
      
      setLoading(false);
    };
    
    checkAuthStatus();
  }, []);
  
  const handleLogin = (authenticated, role) => {
    setIsAuthenticated(authenticated);
    setUserRole(role);
    
    if (authenticated) {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('userRole', role);
    }
  };
  
  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserRole(null);
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('userRole');
  };
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          {/* OTTPlatform route - accessible without login */}
          <Route path="/user" element={<OTTPlatform />} />
          
          {/* Admin login route */}
          <Route 
            path="/admin/login" 
            element={
              isAuthenticated && userRole === 'admin' ? 
                <Navigate to="/admin/dashboard" replace /> : 
                <LoginPage onLogin={handleLogin} />
            } 
          />
          
          {/* Admin dashboard route - protected */}
          <Route 
            path="/admin/dashboard" 
            element={
              isAuthenticated && userRole === 'admin' ? 
                <AdminDashboard onLogout={handleLogout} /> : 
                <Navigate to="/admin/login" replace />
            } 
          />
          
          {/* Default redirect */}
          <Route path="/" element={<Navigate to="/user" replace />} />
          <Route path="*" element={<Navigate to="/user" replace />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;