import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Container,
} from '@mui/material';
import {
  CloudUpload,
  Search,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const HomePage = () => {
  const navigate = useNavigate();

  

  const quickActions = [
    {
      title: 'Upload File',
      description: 'Drag & drop or select files to upload',
      icon: <CloudUpload />,
      action: () => navigate('/documents'),
    },
    {
      title: 'Ask Question',
      description: 'Start querying your knowledge base',
      icon: <Search />,
      action: () => navigate('/chat'),
    },
  ];

  return (
    <Container maxWidth="lg">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {/* Hero Section */}
        <Box textAlign="center" mb={6}>
          <Typography variant="h3" component="h1" gutterBottom>
            Welcome to Bastion Workspace
          </Typography>
        </Box>

        

        {/* Quick Actions */}
        <Box mb={6}>
          <Typography variant="h4" gutterBottom textAlign="center">
            Quick Actions
          </Typography>
          <Grid container spacing={3} justifyContent="center">
            {quickActions.map((action, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                  whileHover={{ y: -5 }}
                >
                  <Card 
                    sx={{ 
                      height: '100%',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      '&:hover': {
                        boxShadow: 4,
                      }
                    }}
                    onClick={action.action}
                  >
                    <CardContent sx={{ textAlign: 'center', py: 3 }}>
                      <Box color="primary.main" mb={2}>
                        {action.icon}
                      </Box>
                      <Typography variant="h6" gutterBottom>
                        {action.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {action.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>
      </motion.div>
    </Container>
  );
};

export default HomePage;
