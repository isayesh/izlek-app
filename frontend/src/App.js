import "@/App.css";
import { useEffect } from "react";
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
import Leaderboard from "@/pages/Leaderboard";
import Profile from "@/pages/Profile";
import Friends from "@/pages/Friends";
import Notifications from "@/pages/Notifications";
import Messages from "@/pages/Messages";

// Use REACT_APP_BACKEND_URL when available, but allow the landing page to render without it.
const backendUrl = process.env.REACT_APP_BACKEND_URL || "";

const trimmedBackendUrl = backendUrl.endsWith("/")
  ? backendUrl.slice(0, -1)
  : backendUrl;

const normalizedBackendUrl = trimmedBackendUrl.endsWith("/api")
  ? trimmedBackendUrl.slice(0, -4)
  : trimmedBackendUrl;

export const API = normalizedBackendUrl ? `${normalizedBackendUrl}/api` : "/api";

function App() {
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    const initialTheme = savedTheme || "light";

    if (initialTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  return (
    <div className="App min-h-screen bg-background text-foreground">
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
            <Route
              path="/leaderboard"
              element={
                <ProtectedRoute>
                  <Leaderboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/friends"
              element={
                <ProtectedRoute>
                  <Friends />
                </ProtectedRoute>
              }
            />
            <Route
              path="/notifications"
              element={
                <ProtectedRoute>
                  <Notifications />
                </ProtectedRoute>
              }
            />
            <Route
              path="/messages"
              element={
                <ProtectedRoute>
                  <Messages />
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
