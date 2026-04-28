import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import axios from "axios";
import { API } from "@/App";
import { Play, Pause, RotateCcw, Send, Users, ArrowLeft, Copy, Check, Clock, MessageCircle, X, Lock } from "lucide-react";
import { startStudySession, updateStudySession, completeStudySession } from "@/lib/studySession";
import { useAuth } from "@/contexts/AuthContext";


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
  const [durationInput, setDurationInput] = useState("25");
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
  const [isOnBreak, setIsOnBreak] = useState(false);
  const autosaveInterval = useRef(null);
  const lastAutosaveSeconds = useRef(0);
  const studyAccumulationStartedAtRef = useRef(null);
  const lastStudySessionSavedAtRef = useRef(null);

  // Chat auto-scroll refs
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const firebaseUid = currentUser?.uid || localStorage.getItem("userId");
  const currentUserId = localStorage.getItem("currentUserId") || firebaseUid;
  const userName = localStorage.getItem("userName");
  const userAvatarUrl = localStorage.getItem("userAvatarUrl") || "";

  const getInitial = (name) => (name || "İ").trim().charAt(0).toUpperCase();

  const isOwner = Boolean(
    room && (room.owner_id === currentUserId || room.owner_id === firebaseUid)
  );
  const chatEnabled = room?.chat_enabled !== false;

  const getParticipantAvatarUrl = (participant) => {
    const resolvedAvatarUrl = participant.avatar_url || (participant.id === currentUserId ? currentProfileAvatarUrl : "");
    return failedParticipantAvatars[participant.id] ? "" : resolvedAvatarUrl;
  };

  const normalizeDurationValue = (rawValue, fallback = 25) => {
    const parsedDuration = Number(rawValue);

    if (!Number.isFinite(parsedDuration) || parsedDuration <= 0) {
      return fallback;
    }

    return Math.min(180, Math.max(1, Math.floor(parsedDuration)));
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
    if (!studyAccumulationStartedAtRef.current) {
      return studySessionBaseSecondsRef.current || totalStudySeconds;
    }
    return studySessionBaseSecondsRef.current + getElapsedSeconds(studyAccumulationStartedAtRef.current);
  };

  const getMessageAvatarUrl = (message) => {
    const resolvedAvatarUrl = message.user_avatar_url || (message.user_id === currentUserId ? currentProfileAvatarUrl : "");
    return failedMessageAvatars[message.id] ? "" : resolvedAvatarUrl;
  };

  const allParticipants = room?.participants || [];
  const HEADER_VISIBLE_PARTICIPANTS = 6;
  const headerVisibleParticipants = allParticipants.slice(0, HEADER_VISIBLE_PARTICIPANTS);
  const headerHiddenParticipantsCount = Math.max(0, allParticipants.length - HEADER_VISIBLE_PARTICIPANTS);
  const headerAvatarSizeClass = allParticipants.length <= 3 ? "h-11 w-11" : allParticipants.length <= 6 ? "h-10 w-10" : "h-9 w-9";

  const renderParticipantItem = (participant, variant = "card") => {
    const participantAvatarUrl = getParticipantAvatarUrl(participant);
    const participantStatusLabel = participant.is_on_break ? "Molada" : isRunning ? "Çalışıyor" : "Hazır";
    const participantStatusPillClasses = participant.is_on_break
      ? "border-yellow-200 bg-yellow-100 text-yellow-700"
      : isRunning
        ? "border-indigo-200 bg-indigo-50 text-indigo-700"
        : "border-slate-200 bg-slate-100 text-slate-700";

    const isPanelVariant = variant === "panel";
    const avatarSizeClass = allParticipants.length <= 4 ? "h-14 w-14" : allParticipants.length <= 8 ? "h-12 w-12" : "h-10 w-10";

    return (
      <div
        key={`${variant}-${participant.id}`}
        className={
          isPanelVariant
            ? "group flex flex-col items-center px-2 py-2.5 text-center transition-colors duration-200 hover:bg-indigo-50/35"
            : "flex items-center gap-3.5 rounded-2xl border border-border/50 bg-background/45 px-4 py-3.5 shadow-[0_18px_38px_-34px_rgba(2,6,23,0.95)] transition-[background-color,border-color,box-shadow] duration-200 hover:border-border/60 hover:bg-white/5 hover:shadow-[0_22px_42px_-34px_rgba(2,6,23,1)]"
        }
        data-testid={`${variant}-participant-${participant.id}`}
      >
        <div
          className={`${isPanelVariant ? `${avatarSizeClass} rounded-full ring-1 ring-border/40` : "h-[52px] w-[52px] rounded-2xl"} flex shrink-0 items-center justify-center overflow-hidden bg-primary/10 text-base font-semibold text-foreground ring-1 ring-primary/20`}
          data-testid={`${variant}-participant-avatar-${participant.id}`}
        >
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
        <div className={isPanelVariant ? "mt-2 min-w-0" : "min-w-0 flex-1"} data-testid={`${variant}-participant-info-${participant.id}`}>
          <p className={`${isPanelVariant ? "text-sm" : "truncate text-sm sm:text-[0.95rem]"} font-semibold text-foreground`} data-testid={`${variant}-participant-name-${participant.id}`}>{participant.name}</p>
          {participant.study_field && (
            <p className={`${isPanelVariant ? "mt-1 line-clamp-1 text-[11px]" : "mt-1 truncate text-xs"} font-medium text-muted-foreground/85`} data-testid={`${variant}-participant-study-field-${participant.id}`}>{participant.study_field}</p>
          )}
        </div>
        <span className={`${isPanelVariant ? "mt-2" : ""} inline-flex shrink-0 items-center rounded-full border px-2.5 py-1 text-xs font-semibold leading-none ${participantStatusPillClasses}`} data-testid={`${variant}-participant-status-badge-${participant.id}`}>
          {participantStatusLabel}
        </span>
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
            const updatedSession = await updateStudySession(studySessionId, totalStudySeconds);
            lastAutosaveSeconds.current = updatedSession?.accumulated_seconds ?? totalStudySeconds;
            lastStudySessionSavedAtRef.current = updatedSession?.last_saved_at || new Date().toISOString();
            console.log(`Auto-saved: ${lastAutosaveSeconds.current} seconds`);
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
      const currentParticipantOnBreak = Boolean(currentParticipant?.is_on_break);
      setIsOnBreak(currentParticipantOnBreak);

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
        const shouldPreserveLocalDuration = !isSyncedTimerRunning && hasLocalFreshDurationOverrideRef.current && !canResumeTimerRef.current;

        if (shouldPreserveLocalDuration) {
          console.log('⏭️ Skipping paused timer sync - local fresh duration override active');
        } else {
          setDuration(syncedDuration);
          setDurationInput(String(syncedDuration));

          if (isSyncedTimerRunning) {
            timerStartingSecondsRef.current = syncedRemainingSeconds;
            timerStartedAtRef.current = res.data.timer_state.started_at;
            studySessionBaseSecondsRef.current = lastAutosaveSeconds.current || totalStudySeconds;
            studyAccumulationStartedAtRef.current = currentParticipantOnBreak
              ? null
              : (lastStudySessionSavedAtRef.current || res.data.timer_state.started_at);

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
            studyAccumulationStartedAtRef.current = null;
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
        setIsOnBreak(Boolean(session.is_on_break));
        lastAutosaveSeconds.current = restoredSeconds;
        studySessionBaseSecondsRef.current = restoredSeconds;
        studyAccumulationStartedAtRef.current = null;
        lastStudySessionSavedAtRef.current = session.last_saved_at || session.started_at || null;
        
        console.log(`✅ Session restored: ${restoredSeconds}s accumulated`);
        console.log('ℹ️ Timer will continue from where you left off on next start');
        console.log('ℹ️ Study progress is preserved for leaderboard');
      }
    } catch (error) {
      console.error('❌ Error checking active session:', error);
    }
  };

  // Resolve scrollable viewport inside chat area
  const getChatViewport = () => {
    const container = chatContainerRef.current;
    if (!container) return null;

    return container.querySelector("[data-radix-scroll-area-viewport]") || container;
  };

  // Check if user is at bottom of chat
  const isUserAtBottom = () => {
    const viewport = getChatViewport();
    if (!viewport) return true;

    const { scrollTop, scrollHeight, clientHeight } = viewport;
    // Consider user at bottom if within 100px of bottom
    return scrollHeight - scrollTop - clientHeight < 100;
  };

  // Scroll to bottom of chat (inside chat viewport only)
  const scrollToBottom = () => {
    const viewport = getChatViewport();
    if (!viewport) return;

    viewport.scrollTop = viewport.scrollHeight;
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

    if (!chatEnabled) {
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

  const handleToggleChatEnabled = async () => {
    if (!isOwner || !room) {
      return;
    }
    const nextChatEnabled = !chatEnabled;
    try {
      const response = await axios.patch(`${API}/rooms/${roomId}/chat`, {
        owner_id: room.owner_id,
        chat_enabled: nextChatEnabled,
      });
      setRoom(response.data);
      // Refresh messages so the newly inserted system status message shows up immediately
      await loadMessages();
      setTimeout(() => scrollToBottom(), 100);
    } catch (error) {
      console.error("Error toggling chat enabled:", error);
    }
  };

  const handleStartTimer = async () => {
    const normalizedDuration = normalizeDurationValue(durationInput, duration);
    const totalSeconds = normalizedDuration * 60;
    const startingSeconds = canResumeTimerRef.current && remainingSeconds > 0 ? remainingSeconds : totalSeconds;
    const startedAt = new Date().toISOString();
    console.log(`🟢 TIMER START: ${normalizedDuration} minutes (${startingSeconds} seconds remaining)`);
    setDuration(normalizedDuration);
    setDurationInput(String(normalizedDuration));
    timerStartingSecondsRef.current = startingSeconds;
    timerStartedAtRef.current = startedAt;
    studySessionBaseSecondsRef.current = totalStudySeconds;
    studyAccumulationStartedAtRef.current = isOnBreak ? null : startedAt;
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
          const sessionBreakState = Boolean(session.is_on_break) || isOnBreak;
          setStudySessionId(session.id);
          setTotalStudySeconds(restoredSeconds);
          setIsOnBreak(sessionBreakState);
          lastAutosaveSeconds.current = restoredSeconds;
          studySessionBaseSecondsRef.current = restoredSeconds;
          studyAccumulationStartedAtRef.current = sessionBreakState ? null : startedAt;
          lastStudySessionSavedAtRef.current = session.last_saved_at || session.started_at || startedAt;
          console.log('✅ Study session ready:', session.id);
        } else {
          // Existing session - continue tracking
          studyAccumulationStartedAtRef.current = isOnBreak ? null : startedAt;
          console.log('✅ Continuing existing session:', studySessionId);
          console.log(`📊 Current accumulated time: ${totalStudySeconds}s`);
          console.log('ℹ️ New timer countdown will ADD to existing accumulated time');
        }
      }
      
      // Update room timer state
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: true,
        duration_minutes: normalizedDuration,
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
    studyAccumulationStartedAtRef.current = null;
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
        const completedSession = await completeStudySession(studySessionId, finalStudySeconds);
        lastStudySessionSavedAtRef.current = completedSession?.last_saved_at || new Date().toISOString();
        console.log(`✅ Study session completed: ${completedSession?.accumulated_seconds ?? finalStudySeconds} seconds`);
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
    studyAccumulationStartedAtRef.current = null;
    
    // Save current progress when pausing
    if (studySessionId && currentStudySeconds > lastAutosaveSeconds.current) {
      try {
        const updatedSession = await updateStudySession(studySessionId, currentStudySeconds);
        lastAutosaveSeconds.current = updatedSession?.accumulated_seconds ?? currentStudySeconds;
        lastStudySessionSavedAtRef.current = updatedSession?.last_saved_at || new Date().toISOString();
        console.log('✅ Progress saved on pause:', lastAutosaveSeconds.current, 'seconds');
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

  const handleToggleBreakMode = async () => {
    if (!roomId || !currentUserId) {
      return;
    }

    const nextIsOnBreak = !isOnBreak;
    const currentStudySeconds = getLiveStudySeconds();
    let persistedStudySeconds = currentStudySeconds;

    try {
      if (nextIsOnBreak) {
        setTotalStudySeconds(currentStudySeconds);
        studySessionBaseSecondsRef.current = currentStudySeconds;
        studyAccumulationStartedAtRef.current = null;

        if (studySessionId && currentStudySeconds > lastAutosaveSeconds.current) {
          const updatedSession = await updateStudySession(studySessionId, currentStudySeconds);
          persistedStudySeconds = updatedSession?.accumulated_seconds ?? currentStudySeconds;
          lastAutosaveSeconds.current = persistedStudySeconds;
          lastStudySessionSavedAtRef.current = updatedSession?.last_saved_at || new Date().toISOString();
          setTotalStudySeconds(persistedStudySeconds);
          studySessionBaseSecondsRef.current = persistedStudySeconds;
        }
      }

      const response = await axios.put(`${API}/rooms/${roomId}/break-mode`, {
        participant_id: currentUserId,
        firebase_uid: firebaseUid,
        is_on_break: nextIsOnBreak
      });

      setRoom(response.data);
      setIsOnBreak(nextIsOnBreak);

      if (nextIsOnBreak) {
        studyAccumulationStartedAtRef.current = null;
      } else if (isRunning) {
        const resumedAt = new Date().toISOString();
        studySessionBaseSecondsRef.current = persistedStudySeconds;
        studyAccumulationStartedAtRef.current = resumedAt;
        setTotalStudySeconds(persistedStudySeconds);
      } else {
        studySessionBaseSecondsRef.current = persistedStudySeconds;
        studyAccumulationStartedAtRef.current = null;
      }
    } catch (error) {
      console.error('❌ Error toggling break mode:', error);
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

  const handleBackToRooms = () => {
    if (window.history.length > 1) {
      navigate(-1);
      return;
    }

    navigate("/dashboard");
  };

  const handleLeaveRoom = async () => {
    await leaveRoom();
    localStorage.removeItem("active_room");
    localStorage.removeItem("currentRoomId");
    navigate("/rooms");
  };

  const handleResetTimer = async () => {
    console.log('🔄 TIMER RESET');
    const normalizedDuration = normalizeDurationValue(durationInput, duration);
    setDuration(normalizedDuration);
    setDurationInput(String(normalizedDuration));
    setIsRunning(false);
    isRunningRef.current = false; // Update ref
    canResumeTimerRef.current = false;
    hasLocalFreshDurationOverrideRef.current = false;
    const totalSeconds = normalizedDuration * 60;
    timerStartingSecondsRef.current = totalSeconds;
    timerStartedAtRef.current = null;
    studySessionBaseSecondsRef.current = 0;
    studyAccumulationStartedAtRef.current = null;
    setRemainingSeconds(totalSeconds);
    setTotalStudySeconds(0);
    
    // Reset study tracking
    lastAutosaveSeconds.current = 0;
    lastStudySessionSavedAtRef.current = null;
    setStudySessionId(null);

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: normalizedDuration,
        remaining_seconds: totalSeconds,
        started_at: null
      });
    } catch (error) {
      console.error("Error resetting timer:", error);
    }
  };

  const handleDurationChange = (e) => {
    const nextDurationInput = e.target.value;
    setDurationInput(nextDurationInput);

    if (nextDurationInput === "") {
      if (!isRunning && !canResumeTimerRef.current) {
        hasLocalFreshDurationOverrideRef.current = true;
      }
      return;
    }

    const parsedDuration = Number(nextDurationInput);
    if (!Number.isFinite(parsedDuration) || parsedDuration <= 0) {
      return;
    }

    if (!isRunning && !canResumeTimerRef.current) {
      const normalizedDuration = normalizeDurationValue(parsedDuration, duration);
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
      <div className="flex min-h-screen items-center justify-center bg-[#F3F5FC]" data-testid="room-loading-state">
        <div className="text-center" data-testid="room-loading-content">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-accent" data-testid="room-loading-spinner"></div>
          <p className="text-muted-foreground" data-testid="room-loading-text">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (room?.error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#F3F5FC]" data-testid="room-error-state">
        <div className="max-w-md rounded-2xl border border-border/55 bg-white/78 p-8 text-center shadow-sm" data-testid="room-error-card">
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
      <div className="flex min-h-screen flex-col items-center justify-center bg-[#F3F5FC] px-6" data-testid="room-not-found-state">
        <p className="mb-4 text-muted-foreground" data-testid="room-not-found-message">Oda bulunamadı</p>
        <Button onClick={() => navigate("/rooms")} data-testid="room-not-found-back-button">Odalara Dön</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F3F5FC] px-4 py-4 sm:px-6 xl:px-8 2xl:px-10 flex flex-col" data-testid="room-page">
      {/* Header */}
      <div className="mb-4 w-full shrink-0" data-testid="room-header-wrapper">
        <div className="rounded-xl border border-slate-200/90 bg-white/88 p-4 shadow-[0_10px_24px_-18px_rgba(79,70,229,0.25)] sm:p-5" data-testid="room-header-card">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                onClick={handleBackToRooms}
                className="h-10 w-10 p-0"
                data-testid="btn-back"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>

              <div className="space-y-1">
                <h1 className="font-display text-2xl font-semibold tracking-tight text-foreground sm:text-3xl" data-testid="room-name">
                  {room.name}
                </h1>
                <div className="flex flex-wrap items-center gap-2" data-testid="room-meta-row">
                  <span className="text-sm font-medium text-muted-foreground" data-testid="room-code-label">Kod:</span>
                  <span className="inline-flex items-center rounded-full border border-slate-200/90 bg-white/78 px-3 py-1 text-sm font-semibold tracking-wide text-foreground" data-testid="room-code-value">
                    {room.code}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyRoomCode}
                    className="h-8 w-8 rounded-full border border-border/60 bg-background/55 p-0 hover:bg-secondary/80"
                    data-testid="btn-copy-code"
                  >
                    {copied ? (
                      <Check className="h-3.5 w-3.5 text-emerald-600 " />
                    ) : (
                      <Copy className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </div>
            </div>

            <div className="flex min-w-0 flex-wrap items-center justify-end gap-2" data-testid="room-header-participants-row">
              {headerVisibleParticipants.map((participant) => {
                const participantAvatarUrl = getParticipantAvatarUrl(participant);
                return (
                  <button
                    key={`header-participant-${participant.id}`}
                    type="button"
                    onClick={() => setShowParticipantsModal(true)}
                    className={`${headerAvatarSizeClass} flex items-center justify-center overflow-hidden rounded-full border border-slate-200/90 bg-white text-xs font-semibold text-foreground shadow-sm transition-colors duration-200 hover:bg-indigo-50/70`}
                    data-testid={`header-participant-avatar-${participant.id}`}
                    title={participant.name}
                  >
                    {participantAvatarUrl ? (
                      <img
                        src={participantAvatarUrl}
                        alt={`${participant.name} avatar`}
                        className="h-full w-full object-cover"
                        data-testid={`header-participant-avatar-image-${participant.id}`}
                        onError={() => setFailedParticipantAvatars((prev) => ({ ...prev, [participant.id]: true }))}
                      />
                    ) : (
                      getInitial(participant.name)
                    )}
                  </button>
                );
              })}

              {headerHiddenParticipantsCount > 0 && (
                <button
                  type="button"
                  onClick={() => setShowParticipantsModal(true)}
                  className={`${headerAvatarSizeClass} inline-flex items-center justify-center rounded-full border border-slate-200/90 bg-white text-xs font-semibold text-muted-foreground shadow-sm transition-colors duration-200 hover:bg-indigo-50/70`}
                  data-testid="header-participants-overflow"
                  title={`${headerHiddenParticipantsCount} kişi daha`}
                >
                  +{headerHiddenParticipantsCount}
                </button>
              )}

              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowParticipantsModal(true)}
                className="h-8 shrink-0 rounded-full border border-slate-200 bg-white/75 px-3 text-xs font-semibold text-muted-foreground hover:bg-indigo-50/65 hover:text-foreground"
                data-testid="participants-show-more-button"
              >
                Tümünü Gör
              </Button>

              <Button
                type="button"
                variant="outline"
                onClick={handleLeaveRoom}
                className="h-9 shrink-0 rounded-lg border-red-200 bg-red-50 px-3 text-xs font-semibold text-red-700 hover:bg-red-100 hover:text-red-800 sm:h-10 sm:px-4 sm:text-sm"
                data-testid="btn-leave-room"
              >
                <span className="hidden sm:inline">Odadan Ayrıl</span>
                <span className="sm:hidden">Ayrıl</span>
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid w-full flex-1 grid-cols-1 min-w-0 gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,0.95fr)] xl:grid-cols-[minmax(0,1.4fr)_minmax(0,0.9fr)]" data-testid="room-main-grid">
        {/* Left: Participants + Timer */}
        <div className="min-w-0" data-testid="room-left-column">
          <Dialog open={showParticipantsModal} onOpenChange={setShowParticipantsModal}>
            <DialogContent className="flex w-[90vw] max-w-xl max-h-[70vh] flex-col overflow-hidden rounded-2xl border border-border/60 bg-card/95 p-0 shadow-[0_28px_60px_-40px_rgba(2,6,23,0.95)] backdrop-blur-sm [&>button]:hidden" data-testid="participants-modal">
              <DialogHeader className="border-b border-border/60 px-6 py-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <DialogTitle className="text-xl font-semibold text-foreground" data-testid="participants-modal-title">
                      Tüm Katılımcılar ({room.participants.length})
                    </DialogTitle>
                    <DialogDescription className="mt-1 text-sm text-muted-foreground" data-testid="participants-modal-description">
                      Odadaki tüm katılımcıları daha rahat bir görünümle inceleyebilirsin.
                    </DialogDescription>
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
              <ScrollArea className="min-h-0 flex-1 px-6 py-5" data-testid="participants-modal-list-scroll-area">
                <div className="space-y-2.5 pr-3" data-testid="participants-modal-list">
                  {allParticipants.map((participant) => renderParticipantItem(participant, "modal"))}
                </div>
              </ScrollArea>
            </DialogContent>
          </Dialog>

          <div className="rounded-[26px] border border-indigo-200 bg-white/95 shadow-[0_30px_60px_-38px_rgba(79,70,229,0.38)]" data-testid="room-left-surface">
            <div className="px-7 pb-8 pt-8 sm:px-8 sm:pb-9">
              <div className="flex min-w-0 flex-col gap-10 text-center" data-testid="timer-card">
                <div className="mx-auto w-fit space-y-1 rounded-lg border border-indigo-100/80 bg-white/72 px-3.5 py-2.5" data-testid="timer-input-group">
                  <label className="block text-[10px] font-semibold uppercase tracking-[0.12em] text-indigo-700/75 md:text-[11px]" data-testid="timer-minutes-label">
                    Çalışma Süresi (dakika)
                  </label>
                  <div className="flex flex-wrap items-center justify-center gap-2.5" data-testid="timer-minutes-row">
                    <Input
                      type="number"
                      min="1"
                      max="180"
                      value={durationInput}
                      onChange={handleDurationChange}
                      disabled={isRunning}
                      className="h-9 w-14 rounded-md border border-indigo-200/80 bg-white text-center text-base font-semibold leading-none text-foreground shadow-none focus:ring-2 focus:ring-ring focus:ring-offset-0 disabled:opacity-60 md:h-10 md:w-16 md:text-lg xl:h-11 xl:w-[4.5rem]"
                      data-testid="input-timer-minutes"
                    />
                    <span className="text-xs font-medium text-indigo-700/80 md:text-sm" data-testid="timer-minutes-unit">dakika</span>
                  </div>
                </div>

                <div className="flex w-full justify-center" data-testid="timer-display-row">
                  <div className="relative mx-auto flex w-full max-w-[620px] min-w-0 items-center justify-center px-5 py-12 md:px-8 md:py-14 xl:max-w-[640px] xl:px-10 xl:py-16" data-testid="timer-display-wrap">
                    <div className="pointer-events-none absolute inset-0 -z-10 rounded-[28px] bg-indigo-100/55 blur-xl" />
                    <div className={`min-w-0 max-w-full overflow-hidden text-ellipsis font-mono text-center text-[clamp(4.2rem,10.2vw,6.1rem)] font-bold leading-none tracking-[0.07em] text-foreground transition-all duration-200 md:text-[clamp(4.8rem,9vw,6.8rem)] xl:text-[clamp(6rem,7.6vw,8.2rem)] ${isOnBreak ? 'scale-[0.985] opacity-[0.22] blur-[1.5px]' : 'scale-100 opacity-100 blur-0'}`} data-testid="timer-display">
                      {formatTime(remainingSeconds)}
                    </div>

                    {isOnBreak && (
                      <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-2xl border border-slate-200/70 bg-white/72 px-5 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] backdrop-blur-sm" data-testid="break-mode-overlay" aria-hidden="true">
                        <div className="pointer-events-none mx-auto flex max-w-[16rem] flex-col items-center justify-center">
                          <div className="text-sm font-semibold leading-tight tracking-tight text-slate-700 sm:text-[0.95rem] ">
                            Moladasın
                          </div>
                          <div className="mt-1.5 text-[11px] font-medium leading-snug text-slate-500 sm:text-xs ">
                            Bu sürede çalışma süren artmaz
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid w-full grid-cols-1 gap-3.5 sm:grid-cols-3" data-testid="timer-controls">
                  {!isRunning ? (
                    <Button
                      onClick={handleStartTimer}
                      size="lg"
                      className="h-12 w-full justify-center rounded-xl border border-indigo-600 bg-indigo-600 px-4 text-sm font-medium text-white shadow-md transition-all duration-200 hover:bg-indigo-700 hover:border-indigo-700 hover:shadow-lg"
                      data-testid="btn-timer-start"
                    >
                      <Play className="h-5 w-5" />
                      Başlat
                    </Button>
                  ) : (
                    <Button
                      onClick={handlePauseTimer}
                      size="lg"
                      className="h-12 w-full justify-center rounded-xl border border-indigo-600 bg-indigo-600 px-4 text-sm font-medium text-white shadow-md transition-all duration-200 hover:bg-indigo-700 hover:border-indigo-700 hover:shadow-lg"
                      data-testid="btn-timer-pause"
                    >
                      <Pause className="h-5 w-5" />
                      Duraklat
                    </Button>
                  )}

                  <Button
                    onClick={handleToggleBreakMode}
                    size="lg"
                    className={`h-12 w-full justify-center rounded-xl border px-4 text-sm font-medium shadow-none transition-all duration-200 ${isOnBreak ? 'border-amber-200 bg-amber-100/65 text-amber-700 hover:bg-amber-100' : 'border-amber-200 bg-amber-50/75 text-amber-700 hover:bg-amber-100/85'}`}
                    data-testid="btn-break-mode-toggle"
                  >
                    {isOnBreak ? "Çalışmaya Dön" : "Mola Ver"}
                  </Button>

                  <Button
                    onClick={handleResetTimer}
                    variant="outline"
                    size="lg"
                    className="h-12 w-full justify-center rounded-xl border border-slate-200/90 bg-white/55 px-4 text-sm font-medium text-muted-foreground shadow-none transition-all duration-200 hover:border-slate-300 hover:bg-white/80 hover:text-foreground"
                    data-testid="btn-timer-reset"
                  >
                    <RotateCcw className="h-5 w-5" />
                    Reset
                  </Button>
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Right: Chat */}
        <Card className="min-w-0 flex h-[560px] flex-col rounded-2xl border border-gray-200/70 bg-white/70 shadow-sm" data-testid="chat-card">
          <CardHeader className="border-b border-slate-200/90 pb-4">
            <div className="flex items-center justify-between gap-3">
              <CardTitle className="flex items-center gap-2 text-xl font-semibold text-foreground" data-testid="chat-title">
                <MessageCircle className="h-5 w-5 text-indigo-600" />
                Sohbet
              </CardTitle>
              {isOwner && (
                <Button
                  onClick={handleToggleChatEnabled}
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-lg border-border/70 bg-background/45 px-3 text-xs font-medium text-foreground hover:bg-background/70"
                  data-testid="btn-toggle-chat-enabled"
                >
                  {chatEnabled ? "Sohbeti Kapat" : "Sohbeti Aç"}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="relative p-0 flex-1 min-h-0">
            <div className="flex h-full min-h-0 flex-col" data-testid="chat-panel">
              <div className="min-h-0 flex-1 overflow-hidden px-5 pt-6 pb-4" data-testid="chat-messages-wrapper">
                <ScrollArea className="h-full pr-3" ref={chatContainerRef} data-testid="chat-scroll-area">
                  <div className="space-y-3" data-testid="messages-list">
                    {messages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/60 bg-background/40 px-6 py-12 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]" data-testid="chat-empty-state">
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
                              <span className="inline-flex rounded-full border border-gray-200/70 bg-gray-100/80 px-3 py-1 text-xs text-gray-600" data-testid={`system-message-content-${message.id}`}>
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
  className={`flex gap-2 ${isOwnMessage ? 'justify-end' : 'justify-start'} ${isGrouped ? 'mt-1.5' : 'mt-4'}`}
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

                            <div className={`${widthClass} ${isOwnMessage ? 'text-right' : ''}`}>
                              {/* Sender name: only show if not grouped (first message of sender) */}
                              {!isGrouped && (
                                <div className={`flex items-center gap-2 mb-1 ${isOwnMessage ? 'justify-end' : ''}`} data-testid={`message-meta-${message.id}`}>
                                  <span className="text-sm font-semibold text-foreground/90" data-testid={`message-user-name-${message.id}`}>{message.user_name}</span>
                                </div>
                              )}

                              <div
                                className={`inline-block rounded-2xl border px-3.5 py-3 shadow-sm ${
                                  isOwnMessage
                                    ? 'border-transparent bg-slate-900 text-slate-50  '
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
              <div className="flex items-center gap-3 border-t border-slate-200/90 bg-white/72 px-5 py-5 backdrop-blur-sm" data-testid="chat-input-row">
                <Input
                  placeholder={!chatEnabled ? "Sohbet kapalı" : "Mesajını yaz..."}
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  disabled={!chatEnabled}
                  className="h-11 border border-gray-200 bg-[#F9FAFB] text-foreground placeholder:text-muted-foreground shadow-none focus-visible:border-indigo-400 focus-visible:ring-2 focus-visible:ring-indigo-100 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-60"
                  data-testid="input-message"
                />
                <Button
                  onClick={sendMessage}
                  disabled={!chatEnabled}
                  className="h-11 rounded-xl bg-[#4F46E5] px-4 text-white shadow-md transition-colors duration-200 hover:bg-[#4338CA] disabled:cursor-not-allowed disabled:opacity-60"
                  data-testid="btn-send-message"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Locked glass overlay when chat is disabled */}
            {!chatEnabled && (
              <div
                className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 rounded-b-2xl bg-background/50 px-6 text-center backdrop-blur-md "
                data-testid="chat-locked-overlay"
                aria-live="polite"
              >
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-background/70 shadow-sm ring-1 ring-border/60 backdrop-blur-sm">
                  <Lock className="h-5 w-5 text-muted-foreground" data-testid="chat-locked-icon" />
                </div>
                <p className="max-w-xs text-base font-medium leading-relaxed text-foreground/90" data-testid="chat-locked-message">
                  Sohbet şu anda oda sahibi tarafından kapatıldı
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

