import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import axios from "axios";
import { API } from "@/App";
import { Play, Pause, RotateCcw, Send, Users, ArrowLeft, Copy, Check, Clock, MessageCircle } from "lucide-react";

export default function RoomPage() {
  const { roomId } = useParams();
  const navigate = useNavigate();
  
  const [room, setRoom] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  // Timer state
  const [timerMinutes, setTimerMinutes] = useState(25);
  const [isRunning, setIsRunning] = useState(false);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const timerInterval = useRef(null);

  // Chat auto-scroll refs
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const currentUserId = localStorage.getItem("currentUserId") || localStorage.getItem("userId");
  const userName = localStorage.getItem("userName");

  useEffect(() => {
    if (!currentUserId || !userName) {
      navigate("/rooms");
      return;
    }
    loadRoom();
    loadMessages();
    
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
  }, [roomId]);

  // Timer effect
  useEffect(() => {
    if (isRunning && remainingSeconds > 0) {
      timerInterval.current = setInterval(() => {
        setRemainingSeconds(prev => {
          if (prev <= 1) {
            setIsRunning(false);
            if (timerInterval.current) {
              clearInterval(timerInterval.current);
            }
            // Timer finished
            alert("⏰ Süre doldu!");
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
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
  }, [isRunning, remainingSeconds]);

  const loadRoom = async () => {
    try {
      const res = await axios.get(`${API}/rooms/${roomId}`);
      setRoom(res.data);
      
      // Sync timer state from room
      if (res.data.timer_state) {
        setTimerMinutes(res.data.timer_state.duration_minutes);
        setIsRunning(res.data.timer_state.is_running);
        
        if (res.data.timer_state.is_running && res.data.timer_state.started_at) {
          // Calculate remaining time
          const startTime = new Date(res.data.timer_state.started_at);
          const now = new Date();
          const elapsed = Math.floor((now - startTime) / 1000);
          const total = res.data.timer_state.duration_minutes * 60;
          const remaining = Math.max(0, total - elapsed);
          setRemainingSeconds(remaining);
          
          if (remaining <= 0) {
            setIsRunning(false);
          }
        } else {
          setRemainingSeconds(res.data.timer_state.remaining_seconds || res.data.timer_state.duration_minutes * 60);
        }
      }
    } catch (error) {
      console.error("Error loading room:", error);
    } finally {
      setLoading(false);
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
    const totalSeconds = timerMinutes * 60;
    setRemainingSeconds(totalSeconds);
    setIsRunning(true);

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: true,
        duration_minutes: timerMinutes,
        remaining_seconds: totalSeconds,
        started_at: new Date().toISOString()
      });
    } catch (error) {
      console.error("Error starting timer:", error);
    }
  };

  const handlePauseTimer = async () => {
    setIsRunning(false);

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: timerMinutes,
        remaining_seconds: remainingSeconds,
        started_at: null
      });
    } catch (error) {
      console.error("Error pausing timer:", error);
    }
  };

  const handleResetTimer = async () => {
    setIsRunning(false);
    const totalSeconds = timerMinutes * 60;
    setRemainingSeconds(totalSeconds);

    try {
      await axios.put(`${API}/rooms/${roomId}/timer`, {
        is_running: false,
        duration_minutes: timerMinutes,
        remaining_seconds: totalSeconds,
        started_at: null
      });
    } catch (error) {
      console.error("Error resetting timer:", error);
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
    return <div className="flex items-center justify-center min-h-screen">Yükleniyor...</div>;
  }

  if (!room) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <p className="mb-4">Oda bulunamadı</p>
        <Button onClick={() => navigate("/rooms")}>Odalara Dön</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => navigate("/rooms")} className="shadow-sm hover:shadow-md transition-shadow" data-testid="btn-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-4xl font-extrabold text-gray-900" data-testid="room-name">
                {room.name}
              </h1>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-sm text-gray-600 font-semibold">Kod: {room.code}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={copyRoomCode}
                  className="h-7 w-7 p-0 hover:bg-gray-100 transition-colors"
                  data-testid="btn-copy-code"
                >
                  {copied ? <Check className="h-3 w-3 text-green-600" /> : <Copy className="h-3 w-3" />}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-8">
        {/* Left: Participants + Timer CENTERPIECE */}
        <div className="space-y-8">
          {/* Participants */}
          <Card className="shadow-md border border-gray-200" data-testid="participants-card">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Users className="h-5 w-5" />
                Katılımcılar ({room.participants.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {room.participants.map(participant => (
                  <div
                    key={participant.id}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors"
                    data-testid={`participant-${participant.id}`}
                  >
                    <div className="w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center font-bold text-sm shadow-sm">
                      {participant.name[0].toUpperCase()}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-sm">{participant.name}</p>
                      {participant.study_field && (
                        <p className="text-xs text-gray-500">{participant.study_field}</p>
                      )}
                    </div>
                    {participant.id === room.owner_id && (
                      <span className="ml-auto text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded font-medium">
                        Oda Sahibi
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Timer - CENTERPIECE */}
          <Card className="border-2 border-indigo-300 shadow-2xl bg-gradient-to-br from-white to-indigo-50" data-testid="timer-card">
            <CardHeader className="pb-4">
              <CardTitle className="text-2xl text-indigo-700 font-extrabold flex items-center gap-2">
                <Clock className="h-7 w-7" />
                Kronometre
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center">
                {/* Timer Input - PRIMARY */}
                <div className="mb-8">
                  <label className="block text-sm font-bold text-gray-700 mb-4">
                    Çalışma Süresi Belirle
                  </label>
                  <div className="flex items-center justify-center gap-4">
                    <Input
                      type="number"
                      min="1"
                      max="180"
                      value={timerMinutes}
                      onChange={(e) => setTimerMinutes(parseInt(e.target.value) || 1)}
                      disabled={isRunning}
                      className="text-center text-3xl font-extrabold h-16 w-28 border-3 border-indigo-400 rounded-xl shadow-md focus:ring-4 focus:ring-indigo-200 transition-all"
                      data-testid="input-timer-minutes"
                    />
                    <span className="text-xl text-gray-700 font-bold">dakika</span>
                  </div>
                </div>
                
                {/* Timer Display - Secondary */}
                <div className="text-5xl font-extrabold text-indigo-600 mb-8 tracking-tight" data-testid="timer-display">
                  {formatTime(remainingSeconds)}
                </div>

                <div className="flex gap-3 justify-center mb-6">
                  {!isRunning ? (
                    <Button onClick={handleStartTimer} size="lg" className="shadow-lg hover:shadow-xl transition-shadow font-semibold" data-testid="btn-timer-start">
                      <Play className="h-5 w-5 mr-2" />
                      Başlat
                    </Button>
                  ) : (
                    <Button onClick={handlePauseTimer} variant="outline" size="lg" className="shadow-md hover:shadow-lg transition-shadow font-semibold" data-testid="btn-timer-pause">
                      <Pause className="h-5 w-5 mr-2" />
                      Duraklat
                    </Button>
                  )}
                  
                  <Button onClick={handleResetTimer} variant="outline" size="lg" className="shadow-md hover:shadow-lg transition-shadow font-semibold" data-testid="btn-timer-reset">
                    <RotateCcw className="h-5 w-5" />
                  </Button>
                </div>

                <p className="text-xs text-gray-500 px-4">
                  Kronometre tüm katılımcılar için senkronize
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Chat */}
        <Card className="md:col-span-2 shadow-lg border border-gray-200" data-testid="chat-card">
          <CardHeader className="pb-4">
            <CardTitle className="text-2xl font-bold flex items-center gap-2">
              <MessageCircle className="h-6 w-6" />
              Sohbet
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Messages */}
              <ScrollArea className="h-[500px] pr-4" ref={chatContainerRef}>
                <div className="space-y-4">
                  {messages.length === 0 ? (
                    <p className="text-center text-gray-500 py-12 text-base">
                      Henüz mesaj yok. İlk mesajı sen gönder! 💬
                    </p>
                  ) : (
                    messages.map((message, index) => {
                      const isOwnMessage = message.user_id === currentUserId;
                      const messageLength = message.content.length;
                      // Adaptive width: short messages get narrower bubbles
                      const widthClass = messageLength < 20 ? 'max-w-[30%]' : messageLength < 50 ? 'max-w-[50%]' : 'max-w-[70%]';
                      
                      // Check if this message is grouped with previous (same sender, consecutive)
                      const prevMessage = index > 0 ? messages[index - 1] : null;
                      const isGrouped = prevMessage && prevMessage.user_id === message.user_id;
                      
                      // Check if this is the last message in a group (next message is from different sender or last message overall)
                      const nextMessage = index < messages.length - 1 ? messages[index + 1] : null;
                      const isLastInGroup = !nextMessage || nextMessage.user_id !== message.user_id;
                      
                      return (
                        <div
                          key={message.id}
                          className={`flex gap-3 ${isOwnMessage ? 'flex-row-reverse' : ''} ${isGrouped ? 'mt-1' : 'mt-4'}`}
                          data-testid={`message-${message.id}`}
                        >
                          {/* Avatar: only show if not grouped (first message of sender) */}
                          {!isGrouped ? (
                            <div className="w-9 h-9 rounded-full bg-primary text-white flex items-center justify-center font-bold flex-shrink-0 text-sm shadow-sm">
                              {message.user_name[0].toUpperCase()}
                            </div>
                          ) : (
                            <div className="w-9 flex-shrink-0"></div>
                          )}
                          
                          <div className={`flex-1 ${widthClass} ${isOwnMessage ? 'text-right' : ''}`}>
                            {/* Sender name: only show if not grouped (first message of sender) */}
                            {!isGrouped && (
                              <div className={`flex items-center gap-2 mb-1 ${isOwnMessage ? 'justify-end' : ''}`}>
                                <span className="text-sm font-semibold">{message.user_name}</span>
                                {message.user_study_field && (
                                  <span className="text-xs text-gray-500">({message.user_study_field})</span>
                                )}
                              </div>
                            )}
                            
                            <div
                              className={`inline-block p-3 rounded-xl shadow-sm ${
                                isOwnMessage
                                  ? 'bg-primary text-white'
                                  : 'bg-gray-100 text-gray-900 border'
                              }`}
                            >
                              <p className="text-sm break-words">{message.content}</p>
                            </div>
                            
                            {/* Timestamp: show only on last message in group for cleaner look */}
                            {isLastInGroup && (
                              <div className={`flex items-center gap-2 mt-1 text-xs text-gray-400 ${isOwnMessage ? 'justify-end' : ''}`}>
                                <span>
                                  {new Date(message.timestamp).toLocaleTimeString('tr-TR', { 
                                    hour: '2-digit', 
                                    minute: '2-digit' 
                                  })}
                                </span>
                                {isOwnMessage && (
                                  <span className="text-xs text-gray-400">• görüldü</span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                  {/* Invisible element to scroll to */}
                  <div ref={chatEndRef} />
                </div>
              </ScrollArea>

              {/* Message Input */}
              <div className="flex gap-3 pt-2">
                <Input
                  placeholder="Mesajını yaz..."
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  className="border-2 shadow-sm focus:shadow-md transition-shadow"
                  data-testid="input-message"
                />
                <Button onClick={sendMessage} className="shadow-md hover:shadow-lg transition-shadow" data-testid="btn-send-message">
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
