import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import axios from "axios";
import { API } from "@/App";
import { Play, Pause, RotateCcw, Send, Users, ArrowLeft, Copy, Check, Clock, MessageCircle, X } from "lucide-react";
import { startStudySession, updateStudySession, completeStudySession } from "@/lib/studySession";
import { useAuth } from "@/contexts/AuthContext";

const MAX_VISIBLE_PARTICIPANTS = 2;

export default function RoomPage() {
  const { roomId } = useParams();
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  
  const [room, setRoom] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [showParticipantsModal, setShowParticipantsModal] = useState(false);
  const [currentProfileAvatarUrl, setCurrentProfileAvatarUrl] = useState(localStorage.getItem("userAvatarUrl") || "");
  const [failedParticipantAvatars, setFailedParticipantAvatars] = useState({});
  const [failedMessageAvatars, setFailedMessageAvatars] = useState({});

  // Timer state
  const [duration, setDuration] = useState(25);
  const [isRunning, setIsRunning] = useState(false);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const [roomAccessChecked, setRoomAccessChecked] = useState(false);
  const timerInterval = useRef(null);
  const isRunningRef = useRef(false); // Track actual running state for callbacks
  const canResumeTimerRef = useRef(false); // Track whether Start should resume remaining time
  const hasLocalFreshDurationOverrideRef = useRef(false); // Keep local fresh-start duration from being overwritten before Start
  const timerStartedAtRef = useRef(null);
  const timerStartingSecondsRef = useRef(0);
  const studySessionBaseSecondsRef = useRef(0);
  const leaveRequestSentRef = useRef(false);
  
  // Study session state
  const [studySessionId, setStudySessionId] = useState(null);
  const [totalStudySeconds, setTotalStudySeconds] = useState(0);
  const autosaveInterval = useRef(null);
  const lastAutosaveSeconds = useRef(0);

  // Chat auto-scroll refs
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const firebaseUid = currentUser?.uid || localStorage.getItem("userId");
  const currentUserId = localStorage.getItem("currentUserId") || firebaseUid;
  const userName = localStorage.getItem("userName");
  const userAvatarUrl = localStorage.getItem("userAvatarUrl") || "";

  const getInitial = (name) => (name || "İ").trim().charAt(0).toUpperCase();

  const getParticipantAvatarUrl = (participant) => {
    const resolvedAvatarUrl = participant.avatar_url || (participant.id === currentUserId ? currentProfileAvatarUrl : "");
    return failedParticipantAvatars[participant.id] ? "" : resolvedAvatarUrl;
  };

  const getElapsedSeconds = (startedAt) => {
    if (!startedAt) return 0;
    return Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  };

  const getLiveRemainingSeconds = () => {
    if (!timerStartedAtRef.current) {
      return timerStartingSecondsRef.current || remainingSeconds;
    }
    return Math.max(0, timerStartingSecondsRef.current - getElapsedSeconds(timerStartedAtRef.current));
  };

  const getLiveStudySeconds = () => {
    if (!timerStartedAtRef.current) {
      return studySessionBaseSecondsRef.current || totalStudySeconds;
    }
    return studySessionBaseSecondsRef.current + getElapsedSeconds(timerStartedAtRef.current);
  };

  const getMessageAvatarUrl = (message) => {
    const resolvedAvatarUrl = message.user_avatar_url || (message.user_id === currentUserId ? currentProfileAvatarUrl : "");
    return failedMessageAvatars[message.id] ? "" : resolvedAvatarUrl;
  };

  const visibleParticipants = room?.participants?.slice(0, MAX_VISIBLE_PARTICIPANTS) || [];
  const hiddenParticipantsCount = Math.max((room?.participants?.length || 0) - MAX_VISIBLE_PARTICIPANTS, 0);

  const renderParticipantItem = (participant, variant = "card") => {
    const participantAvatarUrl = getParticipantAvatarUrl(participant);

    return (
      <div
        key={`${variant}-${participant.id}`}
        className="flex items-center gap-4 rounded-2xl border border-transparent bg-background/70 px-4 py-3 shadow-sm transition-[background-color,box-shadow] duration-200 hover:bg-card hover:shadow-md"
        data-testid={`${variant}-participant-${participant.id}`}
      >
        <div className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-2xl bg-secondary text-base font-semibold text-foreground shadow-sm ring-1 ring-border/60" data-testid={`${variant}-participant-avatar-${participant.id}`}>
          {participantAvatarUrl ? (
            <img
              src={participantAvatarUrl}
              alt={`${participant.name} avatar`}
              className="h-full w-full object-cover"
              data-testid={`${variant}-participant-avatar-image-${participant.id}`}
              onError={() => setFailedParticipantAvatars((prev) => ({ ...prev, [participant.id]: true }))}
            />
          ) : (
            getInitial(participant.name)
          )}
        </div>
        <div className="min-w-0 flex-1" data-testid={`${variant}-participant-info-${participant.id}`}>
          <p className="truncate text-sm font-semibold text-foreground" data-testid={`${variant}-participant-name-${participant.id}`}>{participant.name}</p>
          {participant.study_field && (
            <p className="mt-0.5 text-xs text-muted-foreground" data-testid={`${variant}-participant-study-field-${participant.id}`}>{participant.study_field}</p>
          )}
        </div>
        {participant.id === room.owner_id && (
          <span className="rounded-full border border-border/70 bg-secondary/80 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground" data-testid={`${variant}-participant-owner-badge-${participant.id}`}>
            Sahip
          </span>
        )}
      </div>
    );
  };

  useEffect(() => {
    const loadCurrentProfileAvatar = async () => {
      if (!firebaseUid) return;

      try {
        const response = await axios.get(`${API}/profile`, {
          params: { firebase_uid: firebaseUid }
        });
        const avatarUrl = response.data?.avatar_url || "";
        setCurrentProfileAvatarUrl(avatarUrl);

        if (avatarUrl) {
          localStorage.setItem("userAvatarUrl", avatarUrl);
        } else {
          localStorage.removeItem("userAvatarUrl");
        }
      } catch (error) {
        console.error("Error loading current profile avatar:", error);
      }
    };

    loadCurrentProfileAvatar();
  }, [firebaseUid]);

  useEffect(() => {
    const ensureRoomAccess = async () => {
      if (!firebaseUid) {
        navigate("/rooms");
        return;
      }

      try {
        const response = await axios.get(`${API}/profile`, {
          params: { firebase_uid: firebaseUid }
        });

        if (!response.data?.handle) {
          navigate("/profile");
          return;
        }

        setRoomAccessChecked(true);
      } catch (error) {
        console.error("Error validating room access handle:", error);
        navigate("/profile");
      }
    };

    ensureRoomAccess();
  }, [firebaseUid, navigate]);

  useEffect(() => {
    if (!roomAccessChecked) {
      return;
    }

    if (!currentUserId || !userName) {
      navigate("/rooms");
      return;
    }

    loadRoom();
    loadMessages();
    checkActiveSession(); // Check for existing active session
    
    // Poll for updates every 3 seconds
    const pollInterval = setInterval(() => {
      loadRoom();
      loadMessages();
    }, 3000);

    return () => {
      clearInterval(pollInterval);
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
      }
    };
  }, [roomAccessChecked, roomId, currentUserId, userName, navigate]);

  // Timer effect
  useEffect(() => {
    const updateRunningTimer = () => {
      const nextRemainingSeconds = getLiveRemainingSeconds();
      const nextStudySeconds = getLiveStudySeconds();

      setRemainingSeconds(nextRemainingSeconds);
      setTotalStudySeconds(nextStudySeconds);

      if (nextRemainingSeconds <= 0) {
        setIsRunning(false);
        isRunningRef.current = false;
        if (timerInterval.current) {
          clearInterval(timerInterval.current);
        }
        handleTimerComplete(nextStudySeconds);
        alert("⏰ Süre doldu!");
      }
    };

    if (isRunning && timerStartedAtRef.current && timerStartingSecondsRef.current > 0) {
      updateRunningTimer();
      timerInterval.current = setInterval(updateRunningTimer, 1000);
    } else {
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
      }
    }

    return () => {
      if (timerInterval.current) {
        clearInterval(timerInterval.current);
      }
    };
  }, [isRunning, studySessionId]);
  
  // Autosave effect - save every 60 seconds while timer is running
  useEffect(() => {
    if (isRunning && studySessionId) {
      autosaveInterval.current = setInterval(async () => {
        // Only save if we've accumulated new seconds since last save
        if (totalStudySeconds > lastAutosaveSeconds.current) {
          try {
            await updateStudySession(studySessionId, totalStudySeconds);
            lastAutosaveSeconds.current = totalStudySeconds;
            console.log(`Auto-saved: ${totalStudySeconds} seconds`);
          } catch (error) {
            console.error('Autosave failed:', error);
          }
        }
      }, 60000); // Every 60 seconds
    } else {
      if (autosaveInterval.current) {
        clearInterval(autosaveInterval.current);
      }
    }

    return () => {
      if (autosaveInterval.current) {
        clearInterval(autosaveInterval.current);
      }
    };
  }, [isRunning, studySessionId, totalStudySeconds]);
  
  // Cleanup on unmount - save progress before leaving
  useEffect(() => {
    return () => {
      if (studySessionId && totalStudySeconds > lastAutosaveSeconds.current) {
        // Final save before unmount
        updateStudySession(studySessionId, totalStudySeconds).catch(console.error);
      }
    };
  }, [studySessionId, totalStudySeconds]);

  const loadRoom = async () => {
    console.log('🔄 ROOM FETCH START');
    try {
      const res = await axios.get(`${API}/rooms/${roomId}`);
      console.log('✅ ROOM FETCH SUCCESS:', res.data);
      setRoom(res.data);
      setLoading(false);
      console.log('✅ LOADING: false');

      const currentParticipant = res.data.participants?.find((participant) => participant.id === currentUserId);
      if (currentParticipant?.name) {
        localStorage.setItem("userName", currentParticipant.name);
      }
      if (currentParticipant?.avatar_url) {
        localStorage.setItem("userAvatarUrl", currentParticipant.avatar_url);
      } else if (currentParticipant) {
        localStorage.removeItem("userAvatarUrl");
      }
      
      // CRITICAL: Only sync timer if we're not actively running locally
      // Use ref to avoid stale closure value
      if (!isRunningRef.current && res.data.timer_state) {
        console.log('🔄 Syncing timer state from room (timer not running)');

        const syncedDuration = res.data.timer_state.duration_minutes || 25;
        const syncedRemainingSeconds = res.data.timer_state.remaining_seconds || 0;
        const isSyncedTimerRunning = Boolean(res.data.timer_state.is_running && res.data.timer_state.started_at);

        if (!isSyncedTimerRunning && hasLocalFreshDurationOverrideRef.current && !canResumeTimerRef.current) {
          console.log('⏭️ Skipping paused timer sync - local fresh duration override active');
        } else {
          setDuration(syncedDuration);

          if (isSyncedTimerRunning) {
            timerStartingSecondsRef.current = syncedRemainingSeconds;
            timerStartedAtRef.current = res.data.timer_state.started_at;
            studySessionBaseSecondsRef.current = lastAutosaveSeconds.current || totalStudySeconds;

            const remaining = getLiveRemainingSeconds();
            setRemainingSeconds(remaining);
            setIsRunning(remaining > 0);
            isRunningRef.current = remaining > 0;
            canResumeTimerRef.current = false;
            hasLocalFreshDurationOverrideRef.current = false;
            console.log(`Timer synced from room: ${remaining}s remaining`);
          } else {
            timerStartingSecondsRef.current = syncedRemainingSeconds;
            timerStartedAtRef.current = null;
            setRemainingSeconds(syncedRemainingSeconds);
            setIsRunning(false);
            isRunningRef.current = false;
            canResumeTimerRef.current = syncedRemainingSeconds > 0 && syncedRemainingSeconds < syncedDuration * 60;
            hasLocalFreshDurationOverrideRef.current = false;
            console.log(`Timer synced from paused state: ${syncedRemainingSeconds}s remaining`);
          }
        }
      } else if (isRunningRef.current) {
        console.log('⏭️ Skipping timer sync - timer running locally');
      }
    } catch (error) {
      console.error("❌ ROOM FETCH ERROR:", error);
      console.error("Error details:", error.response?.data);
      setLoading(false);
      console.log('✅ LOADING: false (error case)');
      // Show error state instead of infinite loading
      setRoom({ error: error.message || "Oda yüklenirken hata oluştu" });
    }
  };

  const loadMessages = async () => {
    try {
      const res = await axios.get(`${API}/messages/${roomId}`);
      setMessages(res.data);
    } catch (error) {
      console.error("Error loading messages:", error);
    }
  };

  // Check for active study session on page load/refresh
  const checkActiveSession = async () => {
    if (!firebaseUid) return;
    
    console.log('🔍 Checking for active study session...');
    try {
      // Try to start a session - if one exists, it will return the existing one
      const response = await axios.post(`${API}/study-sessions/start`, {
        firebase_uid: firebaseUid,
        room_id: roomId
      });
      
      const session = response.data;
      console.log('📊 Active session found:', session);
      
      if (session && !session.is_completed) {
        const restoredSeconds = session.accumulated_seconds || 0;
        // Restore session state
        setStudySessionId(session.id);
        setTotalStudySeconds(restoredSeconds);
        lastAutosaveSeconds.current = restoredSeconds;
        studySessionBaseSecondsRef.current = restoredSeconds;
        
        console.log(`✅ Session restored: ${restoredSeconds}s accumulated`);
        console.log('ℹ️ Timer will continue from where you left off on next start');
        console.log('ℹ️ Study progress is preserved for leaderboard');
      }
    } catch (error) {
      console.error('❌ Error checking active session:', error);
    }
  };

  // Check if user is at bottom of chat
  const isUserAtBottom = () => {
    if (!chatContainerRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
    // Consider user at bottom if within 100px of bottom
    return scrollHeight - scrollTop - clientHeight < 100;
  };

  // Scroll to bottom of chat
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Auto-scroll when messages change (only if user was at bottom)
  useEffect(() => {
    if (messages.length > 0 && isUserAtBottom()) {
      scrollToBottom();
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!newMessage.trim()) {
      return;
    }

    try {
      await axios.post(`${API}/messages`, {
        room_id: roomId,
        user_id: currentUserId,
        user_name: userName,
        user_avatar_url: currentProfileAvatarUrl || userAvatarUrl || null,
        user_study_field: null,
        content: newMessage
      });

      setNewMessage("");
      await loadMessages();
      // Always scroll to bottom after sending message
      setTimeout(() => scrollToBottom(), 100);
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  const handleStartTimer = async () => {
    const totalSeconds = duration * 60;
    const startingSeconds = canResumeTimerRef.current && remainingSeconds > 0 ? remainingSeconds : totalSeconds;
    const startedAt = new Date().toISOString();
    console.log(`🟢 TIMER START: ${duration} minutes (${startingSeconds} seconds remaining)`);
    timerStartingSecondsRef.current = startingSeconds;
    timerStartedAtRef.current = startedAt;
    studySessionBaseSecondsRef.current = totalStudySeconds;
    setRemainingSeconds(startingSeconds);
    setIsRunning(true);
    isRunningRef.current = true; // Set ref immediately
    canResumeTimerRef.current = false;
    hasLocalFreshDurationOverrideRef.current = false;

    try {
      // Start or continue study session
      if (firebaseUid) {
        if (!studySessionId) {
          // No existing session - create new one or restore an active one
          const session = await startStudySession(firebaseUid, roomId);
          const restoredSeconds = session.accumulated_seconds || 0;
          setStudySessionId(session.id);
          setTotalStudySeconds(restoredSeconds);
          lastAutosaveSeconds.current = restoredSeconds;
          studySessionBaseSecondsRef.current = restoredSeconds;
          console.log('✅ Study session ready:', session.id);
        } else {
          // Existing session - continue tracking
          console.log('✅ Continuing existing session:', studySessionId);
          console.log(`📊 Current accumulated time: ${totalStudySeconds}s`);
          console.log('ℹ️ New timer countdown will ADD to existing accumulated time');
        }
      }
      
      // Update room timer state
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: true,
        duration_minutes: duration,
        remaining_seconds: startingSeconds,
        started_at: startedAt
      });
      console.log('✅ Room timer state updated on backend');
    } catch (error) {
      console.error("❌ Error starting timer:", error);
    }
  };
  
  const handleTimerComplete = async (completedStudySeconds = getLiveStudySeconds()) => {
    console.log('✅ TIMER COMPLETED');
    const finalStudySeconds = Math.max(0, completedStudySeconds);
    setRemainingSeconds(0);
    setTotalStudySeconds(finalStudySeconds);
    timerStartingSecondsRef.current = 0;
    timerStartedAtRef.current = null;
    studySessionBaseSecondsRef.current = finalStudySeconds;
    canResumeTimerRef.current = false;
    hasLocalFreshDurationOverrideRef.current = false;

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: duration,
        remaining_seconds: 0,
        started_at: null
      });
    } catch (error) {
      console.error('❌ Error syncing completed timer state:', error);
    }

    // Complete the study session
    if (studySessionId && firebaseUid) {
      try {
        await completeStudySession(studySessionId, finalStudySeconds);
        console.log(`✅ Study session completed: ${finalStudySeconds} seconds`);
        setStudySessionId(null);
      } catch (error) {
        console.error('❌ Error completing study session:', error);
      }
    }
  };

  const handlePauseTimer = async () => {
    console.log('⏸️ TIMER PAUSED');
    const currentRemainingSeconds = getLiveRemainingSeconds();
    const currentStudySeconds = getLiveStudySeconds();

    setIsRunning(false);
    setRemainingSeconds(currentRemainingSeconds);
    setTotalStudySeconds(currentStudySeconds);
    isRunningRef.current = false; // Update ref
    canResumeTimerRef.current = true;
    hasLocalFreshDurationOverrideRef.current = false;
    timerStartingSecondsRef.current = currentRemainingSeconds;
    timerStartedAtRef.current = null;
    studySessionBaseSecondsRef.current = currentStudySeconds;
    
    // Save current progress when pausing
    if (studySessionId && currentStudySeconds > lastAutosaveSeconds.current) {
      try {
        await updateStudySession(studySessionId, currentStudySeconds);
        lastAutosaveSeconds.current = currentStudySeconds;
        console.log('✅ Progress saved on pause:', currentStudySeconds, 'seconds');
      } catch (error) {
        console.error('❌ Error saving progress on pause:', error);
      }
    }

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: duration,
        remaining_seconds: currentRemainingSeconds,
        started_at: null
      });
    } catch (error) {
      console.error("Error pausing timer:", error);
    }
  };

  const leaveRoom = async () => {
    if (leaveRequestSentRef.current || !roomId || !currentUserId) {
      return;
    }

    leaveRequestSentRef.current = true;

    try {
      await axios.post(`${API}/rooms/${roomId}/leave`, {
        user_id: currentUserId,
        user_name: userName
      });
    } catch (error) {
      console.error("Error leaving room:", error);
      leaveRequestSentRef.current = false;
    }
  };

  const handleBackToRooms = async () => {
    await leaveRoom();
    navigate("/rooms");
  };

  const handleResetTimer = async () => {
    console.log('🔄 TIMER RESET');
    setIsRunning(false);
    isRunningRef.current = false; // Update ref
    canResumeTimerRef.current = false;
    hasLocalFreshDurationOverrideRef.current = false;
    const totalSeconds = duration * 60;
    timerStartingSecondsRef.current = totalSeconds;
    timerStartedAtRef.current = null;
    studySessionBaseSecondsRef.current = 0;
    setRemainingSeconds(totalSeconds);
    setTotalStudySeconds(0);
    
    // Reset study tracking
    lastAutosaveSeconds.current = 0;
    setStudySessionId(null);

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: duration,
        remaining_seconds: totalSeconds,
        started_at: null
      });
    } catch (error) {
      console.error("Error resetting timer:", error);
    }
  };

  const handleDurationChange = (e) => {
    const nextDuration = Number(e.target.value);
    const normalizedDuration = Number.isFinite(nextDuration) && nextDuration > 0 ? nextDuration : 25;
    setDuration(normalizedDuration);

    if (!isRunning && !canResumeTimerRef.current) {
      const nextTotalSeconds = normalizedDuration * 60;
      hasLocalFreshDurationOverrideRef.current = true;
      timerStartingSecondsRef.current = nextTotalSeconds;
      setRemainingSeconds(nextTotalSeconds);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const copyRoomCode = () => {
    navigator.clipboard.writeText(room.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background" data-testid="room-loading-state">
        <div className="text-center" data-testid="room-loading-content">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-accent" data-testid="room-loading-spinner"></div>
          <p className="text-muted-foreground" data-testid="room-loading-text">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (room?.error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background" data-testid="room-error-state">
        <div className="max-w-md rounded-2xl border border-destructive/20 bg-card p-8 text-center shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)]" data-testid="room-error-card">
          <h2 className="mb-4 text-2xl font-semibold text-destructive" data-testid="room-error-title">Oda Yüklenemedi</h2>
          <p className="mb-6 text-muted-foreground" data-testid="room-error-message">{room.error}</p>
          <Button onClick={() => navigate('/rooms')} className="w-full" data-testid="room-error-back-button">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Odalar Sayfasına Dön
          </Button>
        </div>
      </div>
    );
  }

  if (!room) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-background px-6" data-testid="room-not-found-state">
        <p className="mb-4 text-muted-foreground" data-testid="room-not-found-message">Oda bulunamadı</p>
        <Button onClick={() => navigate("/rooms")} data-testid="room-not-found-back-button">Odalara Dön</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background px-4 py-6 sm:px-6 lg:px-10 xl:px-12 flex h-screen flex-col overflow-hidden" data-testid="room-page">
      {/* Header */}
      <div className="mb-8 w-full shrink-0" data-testid="room-header-wrapper">
        <div className="rounded-2xl border border-border/70 bg-card/95 p-5 sm:p-6 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)]" data-testid="room-header-card">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                onClick={handleBackToRooms}
                className="h-10 w-10 p-0"
                data-testid="btn-back"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>

              <div className="space-y-2">
                <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-5xl lg:text-6xl" data-testid="room-name">
                  {room.name}
                </h1>
                <div className="flex flex-wrap items-center gap-2" data-testid="room-meta-row">
                  <span className="text-sm font-medium text-muted-foreground" data-testid="room-code-label">Kod:</span>
                  <span className="inline-flex items-center rounded-full border border-border/70 bg-background/70 px-3 py-1 text-sm font-semibold tracking-wide text-foreground" data-testid="room-code-value">
                    {room.code}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyRoomCode}
                    className="h-8 w-8 rounded-full border border-border/70 bg-background/80 p-0 hover:bg-secondary"
                    data-testid="btn-copy-code"
                  >
                    {copied ? (
                      <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <Copy className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid w-full flex-1 min-h-0 gap-6 overflow-y-auto lg:grid-cols-3 lg:overflow-hidden" data-testid="room-main-grid">
        {/* Left: Participants + Timer */}
        <div className="space-y-6 min-h-0 lg:flex lg:h-full lg:flex-col lg:overflow-hidden" data-testid="room-left-column">
          {/* Participants */}
          <Card className="overflow-hidden rounded-2xl border border-border/70 bg-card/95 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)] lg:shrink-0" data-testid="participants-card">
            <CardHeader className="border-b border-border/60 pb-3">
              <CardTitle className="flex items-center gap-2 text-xl font-semibold text-foreground" data-testid="participants-title">
                <Users className="h-5 w-5 text-accent" />
                Katılımcılar ({room.participants.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pb-4 pt-4">
              <div className="space-y-3" data-testid="participants-list">
                {visibleParticipants.map((participant) => renderParticipantItem(participant, "card"))}
              </div>

              {hiddenParticipantsCount > 0 && (
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => setShowParticipantsModal(true)}
                  className="h-auto w-full justify-start rounded-2xl border border-dashed border-border/70 px-4 py-3 text-sm font-medium text-muted-foreground hover:bg-background/80"
                  data-testid="participants-show-more-button"
                >
                  +{hiddenParticipantsCount} kişi daha · Tümünü gör
                </Button>
              )}
            </CardContent>
          </Card>

          <Dialog open={showParticipantsModal} onOpenChange={setShowParticipantsModal}>
            <DialogContent className="max-w-xl rounded-2xl border border-border/70 bg-card p-0 shadow-[0_20px_50px_-30px_rgba(15,23,42,0.45)] [&>button]:hidden" data-testid="participants-modal">
              <DialogHeader className="border-b border-border/60 px-6 py-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <DialogTitle className="text-xl font-semibold text-foreground" data-testid="participants-modal-title">
                      Tüm Katılımcılar ({room.participants.length})
                    </DialogTitle>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowParticipantsModal(false)}
                    className="shrink-0 rounded-xl"
                    data-testid="participants-modal-close-button"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </DialogHeader>
              <div className="max-h-[70vh] space-y-3 overflow-y-auto px-6 py-5" data-testid="participants-modal-list">
                {room.participants.map((participant) => renderParticipantItem(participant, "modal"))}
              </div>
            </DialogContent>
          </Dialog>

          {/* Timer */}
          <Card className="rounded-2xl border border-border/70 bg-card/95 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)] lg:shrink-0" data-testid="timer-card">
            <CardHeader className="border-b border-border/60 pb-3">
              <CardTitle className="flex items-center gap-2 text-xl font-semibold text-foreground" data-testid="timer-title">
                <Clock className="h-5 w-5 text-accent" />
                Kronometre
              </CardTitle>
            </CardHeader>
            <CardContent className="px-6 pb-6 pt-6">
              <div className="space-y-6 text-center" data-testid="timer-panel">
                {/* Timer Input */}
                <div className="space-y-3.5" data-testid="timer-input-group">
                  <label className="block text-sm font-medium tracking-tight text-foreground/90" data-testid="timer-minutes-label">
                    Çalışma Süresi (dakika)
                  </label>
                  <div className="flex items-center justify-center gap-3" data-testid="timer-minutes-row">
                    <Input
                      type="number"
                      min="1"
                      max="180"
                      value={duration}
                      onChange={handleDurationChange}
                      disabled={isRunning}
                      className="h-14 w-28 rounded-2xl border border-border/70 bg-background/80 text-center text-2xl font-semibold leading-none text-foreground shadow-sm focus:ring-2 focus:ring-ring focus:ring-offset-0 disabled:opacity-60"
                      data-testid="input-timer-minutes"
                    />
                    <span className="min-w-[62px] text-left text-base font-medium tracking-tight text-muted-foreground" data-testid="timer-minutes-unit">dakika</span>
                  </div>
                </div>

                {/* Timer Display */}
                <div className="mx-auto flex w-fit min-w-[228px] items-center justify-center rounded-2xl border border-border/70 bg-background/70 px-8 py-5.5 shadow-sm" data-testid="timer-display-wrap">
                  <div className="font-mono text-center text-[3.5rem] font-semibold leading-none tracking-tight text-foreground sm:text-[4.25rem]" data-testid="timer-display">
                    {formatTime(remainingSeconds)}
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-center gap-3.5 pt-1" data-testid="timer-controls">
                  {!isRunning ? (
                    <Button
                      onClick={handleStartTimer}
                      size="lg"
                      className="min-w-[148px] justify-center px-8 rounded-xl"
                      data-testid="btn-timer-start"
                    >
                      <Play className="h-5 w-5 mr-2" />
                      Başlat
                    </Button>
                  ) : (
                    <Button
                      onClick={handlePauseTimer}
                      variant="outline"
                      size="lg"
                      className="min-w-[148px] justify-center px-8 rounded-xl"
                      data-testid="btn-timer-pause"
                    >
                      <Pause className="h-5 w-5 mr-2" />
                      Duraklat
                    </Button>
                  )}

                  <Button
                    onClick={handleResetTimer}
                    variant="outline"
                    size="lg"
                    className="min-w-[56px] rounded-xl"
                    data-testid="btn-timer-reset"
                  >
                    <RotateCcw className="h-5 w-5" />
                  </Button>
                </div>

                <p className="inline-flex items-center rounded-full border border-border/70 bg-secondary/80 px-3 py-1 text-xs text-muted-foreground" data-testid="timer-sync-note">
                  Kronometre tüm katılımcılar için senkronize
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Chat */}
        <Card className="lg:col-span-2 rounded-2xl border border-border/70 bg-card/95 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)] flex min-h-0 flex-col" data-testid="chat-card">
          <CardHeader className="border-b border-border/60 pb-4">
            <CardTitle className="flex items-center gap-2 text-xl font-semibold text-foreground" data-testid="chat-title">
              <MessageCircle className="h-5 w-5 text-accent" />
              Sohbet
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 flex-1 min-h-0">
            <div className="flex h-[560px] min-h-0 flex-col lg:h-full" data-testid="chat-panel">
              <div className="min-h-0 flex-1 px-5 pt-5" data-testid="chat-messages-wrapper">
                <ScrollArea className="h-full pr-3" ref={chatContainerRef} data-testid="chat-scroll-area">
                  <div className="space-y-3" data-testid="messages-list">
                    {messages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-12 text-center" data-testid="chat-empty-state">
                        <p className="text-base font-semibold text-foreground">Henüz mesaj yok</p>
                        <p className="mt-2 text-sm text-muted-foreground">İlk mesajı göndererek odadaki akışı başlat.</p>
                      </div>
                    ) : (
                      messages.map((message, index) => {
                        const isSystemMessage = message.user_id === "system";
                        const isOwnMessage = message.user_id === currentUserId;
                        const messageAvatarUrl = getMessageAvatarUrl(message);
                        const messageLength = message.content.length;
                        const widthClass = messageLength < 20 ? 'max-w-[45%]' : messageLength < 50 ? 'max-w-[65%]' : 'max-w-[85%]';

                        if (isSystemMessage) {
                          return (
                            <div key={message.id} className="py-2 text-center" data-testid={`system-message-${message.id}`}>
                              <span className="inline-flex rounded-full border border-border/70 bg-secondary/80 px-3 py-1 text-xs text-muted-foreground" data-testid={`system-message-content-${message.id}`}>
                                {message.content}
                              </span>
                            </div>
                          );
                        }

                        const prevMessage = index > 0 ? messages[index - 1] : null;
                        const isGrouped = prevMessage && prevMessage.user_id === message.user_id;

                        const nextMessage = index < messages.length - 1 ? messages[index + 1] : null;
                        const isLastInGroup = !nextMessage || nextMessage.user_id !== message.user_id;

                        return (
                          <div
                            key={message.id}
                            className={`flex gap-2 ${isOwnMessage ? 'flex-row-reverse' : ''} ${isGrouped ? 'mt-1.5' : 'mt-4'}`}
                            data-testid={`message-${message.id}`}
                          >
                            {!isGrouped ? (
                              <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-xl bg-secondary text-xs font-semibold text-foreground shadow-sm ring-1 ring-border/60" data-testid={`message-avatar-${message.id}`}>
                                {messageAvatarUrl ? (
                                  <img
                                    src={messageAvatarUrl}
                                    alt={`${message.user_name} avatar`}
                                    className="h-full w-full object-cover"
                                    data-testid={`message-avatar-image-${message.id}`}
                                    onError={() => setFailedMessageAvatars((prev) => ({ ...prev, [message.id]: true }))}
                                  />
                                ) : (
                                  getInitial(message.user_name)
                                )}
                              </div>
                            ) : (
                              <div className="w-8 flex-shrink-0"></div>
                            )}

                            <div className={`flex-1 ${widthClass} ${isOwnMessage ? 'text-right' : ''}`}>
                              {/* Sender name: only show if not grouped (first message of sender) */}
                              {!isGrouped && (
                                <div className={`flex items-center gap-2 mb-1 ${isOwnMessage ? 'justify-end' : ''}`} data-testid={`message-meta-${message.id}`}>
                                  <span className="text-sm font-semibold text-foreground/90" data-testid={`message-user-name-${message.id}`}>{message.user_name}</span>
                                  {message.user_study_field && (
                                    <span className="text-xs text-muted-foreground" data-testid={`message-user-study-field-${message.id}`}>
                                      ({message.user_study_field})
                                    </span>
                                  )}
                                </div>
                              )}

                              <div
                                className={`inline-block rounded-2xl border px-3.5 py-3 shadow-sm ${
                                  isOwnMessage
                                    ? 'border-transparent bg-slate-900 text-slate-50 dark:bg-slate-100 dark:text-slate-950'
                                    : 'border-border/60 bg-secondary/90 text-foreground'
                                }`}
                                data-testid={`message-bubble-${message.id}`}
                              >
                                <p className="text-sm break-words" data-testid={`message-content-${message.id}`}>{message.content}</p>
                              </div>

                              {/* Timestamp: show only on last message in group for cleaner look */}
                              {isLastInGroup && (
                                <div className={`flex items-center gap-2 mt-1 text-xs text-muted-foreground ${isOwnMessage ? 'justify-end' : ''}`} data-testid={`message-timestamp-row-${message.id}`}>
                                  <span data-testid={`message-time-${message.id}`}>
                                    {new Date(message.timestamp).toLocaleTimeString('tr-TR', {
                                      hour: '2-digit',
                                      minute: '2-digit'
                                    })}
                                  </span>
                                  {isOwnMessage && (
                                    <span className="text-xs text-muted-foreground" data-testid={`message-status-${message.id}`}>• görüldü</span>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })
                    )}
                    {/* Invisible element to scroll to */}
                    <div ref={chatEndRef} data-testid="chat-scroll-anchor" />
                  </div>
                </ScrollArea>
              </div>

              {/* Message Input */}
              <div className="flex items-center gap-3 border-t border-border/60 bg-background/70 px-5 py-4" data-testid="chat-input-row">
                <Input
                  placeholder="Mesajını yaz..."
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  className="h-11 border border-border/70 bg-background/80 text-foreground placeholder:text-muted-foreground shadow-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0"
                  data-testid="input-message"
                />
                <Button onClick={sendMessage} className="h-11 px-4 rounded-xl" data-testid="btn-send-message">
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
