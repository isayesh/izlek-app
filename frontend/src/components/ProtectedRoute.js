import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export default function ProtectedRoute({ children }) {
  const { currentUser, firebaseConfigured } = useAuth();

  // If Firebase is not configured, show error message
  if (!firebaseConfigured) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Yapılandırma Hatası</h2>
          <p className="text-gray-600">Firebase yapılandırması eksik. Lütfen yöneticiyle iletişime geçin.</p>
        </div>
      </div>
    );
  }

  // If user is not logged in, redirect to login
  if (!currentUser) {
    return <Navigate to="/login" replace />;
  }

  return children;
}