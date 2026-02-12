import "@/App.css";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";

// Pages
import LandingPage from "@/pages/LandingPage";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import ProgramCreation from "@/pages/ProgramCreation";
import Dashboard from "@/pages/Dashboard";
import Rooms from "@/pages/Rooms";
import RoomPage from "@/pages/RoomPage";
import NetTracking from "@/pages/NetTracking";

// Use REACT_APP_BACKEND_URL from environment for API calls
// Falls back to relative path if not set (for local development)
export const API = process.env.REACT_APP_BACKEND_URL 
  ? `${process.env.REACT_APP_BACKEND_URL}/api`
  : "/api";

function App() {
  return (
    <div className="App min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            
            {/* Protected Routes */}
            <Route 
              path="/program/create" 
              element={
                <ProtectedRoute>
                  <ProgramCreation />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/dashboard" 
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/rooms" 
              element={
                <ProtectedRoute>
                  <Rooms />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/room/:roomId" 
              element={
                <ProtectedRoute>
                  <RoomPage />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/net-tracking" 
              element={
                <ProtectedRoute>
                  <NetTracking />
                </ProtectedRoute>
              } 
            />
            
            {/* Catch all - redirect to landing */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </div>
  );
}

export default App;