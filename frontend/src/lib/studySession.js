import axios from 'axios';

const backendUrl = process.env.REACT_APP_BACKEND_URL || '';

const trimmedBackendUrl = backendUrl.endsWith('/')
  ? backendUrl.slice(0, -1)
  : backendUrl;

const API = trimmedBackendUrl.endsWith('/api')
  ? trimmedBackendUrl
  : trimmedBackendUrl
    ? `${trimmedBackendUrl}/api`
    : '/api';

/**
 * Study session management utilities for timer persistence
 */

export const startStudySession = async (firebaseUid, roomId) => {
  try {
    const response = await axios.post(`${API}/study-sessions/start`, {
      firebase_uid: firebaseUid,
      room_id: roomId
    });
    return response.data;
  } catch (error) {
    console.error('Error starting study session:', error);
    throw error;
  }
};

export const updateStudySession = async (sessionId, accumulatedSeconds) => {
  try {
    const response = await axios.put(`${API}/study-sessions/${sessionId}/update`, {
      accumulated_seconds: accumulatedSeconds
    });
    return response.data;
  } catch (error) {
    console.error('Error updating study session:', error);
    throw error;
  }
};

export const completeStudySession = async (sessionId, accumulatedSeconds) => {
  try {
    const response = await axios.put(`${API}/study-sessions/${sessionId}/complete`, {
      accumulated_seconds: accumulatedSeconds
    });
    return response.data;
  } catch (error) {
    console.error('Error completing study session:', error);
    throw error;
  }
};
