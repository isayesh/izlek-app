import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Home, MessageSquare, Send, Users } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAuth } from "@/contexts/AuthContext";
import { API } from "@/App";
import { formatPublicHandle, getAvatarFallback, getPublicUsername } from "@/lib/publicProfile";

const ROOM_INVITE_MESSAGE_TYPE = "room_invite";

const formatMessageTime = (timestamp) => new Date(timestamp).toLocaleTimeString("tr-TR", {
  hour: "2-digit",
  minute: "2-digit",
});

const formatSidebarTime = (timestamp) => {
  if (!timestamp) {
    return "";
  }

  const messageDate = new Date(timestamp);
  const now = new Date();
  const isToday = messageDate.toDateString() === now.toDateString();

  return isToday
    ? messageDate.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })
    : messageDate.toLocaleDateString("tr-TR", { day: "2-digit", month: "short" });
};

const getDirectMessagePreview = (messageLike) => {
  if (!messageLike) {
    return "";
  }

  if (messageLike.message_type === ROOM_INVITE_MESSAGE_TYPE) {
    const roomName = messageLike.room_invite?.room_name?.trim();
    return roomName ? `"${roomName}" odasına davet` : "Oda daveti";
  }

  return (messageLike.message || messageLike.last_message || "").trim();
};

const getRoomInviteCopy = (message, isOwnMessage) => {
  const roomName = message.room_invite?.room_name?.trim();
  const inviterName = message.room_invite?.inviter_name?.trim();

  if (isOwnMessage) {
    return roomName ? `"${roomName}" odası için davet gönderdin.` : "Bu konuşmaya bir oda daveti gönderdin.";
  }

  if (roomName && inviterName) {
    return `${inviterName} seni "${roomName}" odasına çağırıyor.`;
  }

  if (roomName) {
    return `Seni "${roomName}" odasına çağırıyor.`;
  }

  if (inviterName) {
    return `${inviterName} seni bir çalışma odasına çağırıyor.`;
  }

  return "Seni bir çalışma odasına çağırıyor.";
};

export default function Messages() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [friends, setFriends] = useState([]);
  const [friendsLoading, setFriendsLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [sendingInvite, setSendingInvite] = useState(false);
  const [roomInviteLoading, setRoomInviteLoading] = useState(false);
  const [selectedFriendId, setSelectedFriendId] = useState("");
  const [draftMessage, setDraftMessage] = useState("");
  const [activeMessages, setActiveMessages] = useState([]);
  const [conversationSummaries, setConversationSummaries] = useState({});
  const [activeRoomInviteTarget, setActiveRoomInviteTarget] = useState(null);
  const [error, setError] = useState("");

  const authHeaders = useMemo(() => (
    currentUser?.uid ? { "X-Firebase-UID": currentUser.uid } : {}
  ), [currentUser]);

  const currentRoomParticipantId = useMemo(
    () => localStorage.getItem("currentUserId") || currentUser?.uid || "",
    [currentUser]
  );

  const loadFriends = useCallback(async () => {
    if (!currentUser?.uid) {
      setFriendsLoading(false);
      return;
    }

    try {
      setFriendsLoading(true);
      const response = await axios.get(`${API}/friends`, { headers: authHeaders });
      setFriends(Array.isArray(response.data) ? response.data : []);
    } catch (loadError) {
      console.error("Error loading friends for messages:", loadError);
      setError("Arkadaş listesi yüklenirken bir hata oluştu.");
    } finally {
      setFriendsLoading(false);
    }
  }, [authHeaders, currentUser]);

  const loadMessages = useCallback(async (friendProfileId) => {
    if (!currentUser?.uid || !friendProfileId) {
      setActiveMessages([]);
      setMessagesLoading(false);
      return;
    }

    try {
      setMessagesLoading(true);
      setError("");
      const response = await axios.get(`${API}/messages/direct/${friendProfileId}`, { headers: authHeaders });
      setActiveMessages(Array.isArray(response.data) ? response.data : []);
      setConversationSummaries((previousSummaries) => ({
        ...previousSummaries,
        [friendProfileId]: {
          ...(previousSummaries[friendProfileId] || {}),
          unread_count: 0,
        },
      }));
    } catch (loadError) {
      console.error("Error loading direct messages:", loadError);
      setError(loadError.response?.data?.detail || "Mesajlar yüklenirken bir hata oluştu.");
      setActiveMessages([]);
    } finally {
      setMessagesLoading(false);
    }
  }, [authHeaders, currentUser]);

  const loadConversationSummaries = useCallback(async () => {
    if (!currentUser?.uid) {
      setConversationSummaries({});
      return;
    }

    try {
      const response = await axios.get(`${API}/messages/direct/sidebar-summary`, { headers: authHeaders });
      setConversationSummaries(response.data?.summaries || {});
    } catch (loadError) {
      console.error("Error loading direct message conversation summaries:", loadError);
    }
  }, [authHeaders, currentUser]);

  const loadActiveRoomInviteTarget = useCallback(async () => {
    if (!currentUser?.uid) {
      setActiveRoomInviteTarget(null);
      setRoomInviteLoading(false);
      return;
    }

    const storedRoomId = localStorage.getItem("currentRoomId");
    if (!storedRoomId) {
      setActiveRoomInviteTarget(null);
      setRoomInviteLoading(false);
      return;
    }

    try {
      setRoomInviteLoading(true);
      const response = await axios.get(`${API}/rooms/${storedRoomId}`);
      const room = response.data;
      const participants = Array.isArray(room?.participants) ? room.participants : [];
      const isParticipant = participants.some(
        (participant) => participant?.id === currentRoomParticipantId || participant?.id === currentUser.uid
      );

      if (!room?.id || !isParticipant) {
        setActiveRoomInviteTarget(null);
        return;
      }

      setActiveRoomInviteTarget({
        room_id: room.id,
        room_name: room.name || "",
      });
    } catch (loadError) {
      console.error("Error resolving active room invite target:", loadError);
      setActiveRoomInviteTarget(null);
    } finally {
      setRoomInviteLoading(false);
    }
  }, [currentRoomParticipantId, currentUser]);

  useEffect(() => {
    loadFriends();
    loadConversationSummaries();
    loadActiveRoomInviteTarget();
  }, [loadFriends, loadConversationSummaries, loadActiveRoomInviteTarget]);

  useEffect(() => {
    if (!selectedFriendId) {
      setActiveMessages([]);
      setMessagesLoading(false);
      return;
    }

    loadMessages(selectedFriendId);
  }, [loadMessages, selectedFriendId]);

  const selectedFriend = useMemo(
    () => friends.find((friend) => friend.profile_id === selectedFriendId) || null,
    [friends, selectedFriendId]
  );
  const selectedFriendHandle = useMemo(() => {
    if (!selectedFriend?.handle) {
      return "";
    }

    return formatPublicHandle(selectedFriend.handle_display || selectedFriend.handle);
  }, [selectedFriend]);

  const handleSendMessage = async (event) => {
    event.preventDefault();

    const trimmedMessage = draftMessage.trim();
    if (!selectedFriendId || !trimmedMessage) {
      return;
    }

    try {
      setSendingMessage(true);
      setError("");
      const response = await axios.post(
        `${API}/messages/direct`,
        {
          receiver_profile_id: selectedFriendId,
          message: trimmedMessage,
        },
        { headers: authHeaders }
      );

      setActiveMessages((previousMessages) => [...previousMessages, response.data]);
      setConversationSummaries((previousSummaries) => ({
        ...previousSummaries,
        [selectedFriendId]: {
          ...(previousSummaries[selectedFriendId] || {}),
          last_message: getDirectMessagePreview(response.data),
          last_message_at: response.data?.created_at || new Date().toISOString(),
          unread_count: 0,
        },
      }));
      setDraftMessage("");
    } catch (sendError) {
      console.error("Error sending direct message:", sendError);
      setError(sendError.response?.data?.detail || "Mesaj gönderilirken bir hata oluştu.");
    } finally {
      setSendingMessage(false);
    }
  };

  const handleSendRoomInvite = async () => {
    if (!selectedFriendId) {
      return;
    }

    if (!activeRoomInviteTarget?.room_id) {
      setError("Davet göndermek için önce aktif olduğun bir room seçili olmalı.");
      return;
    }

    try {
      setSendingInvite(true);
      setError("");
      const response = await axios.post(
        `${API}/messages/direct`,
        {
          receiver_profile_id: selectedFriendId,
          message_type: ROOM_INVITE_MESSAGE_TYPE,
          room_invite: {
            room_id: activeRoomInviteTarget.room_id,
          },
        },
        { headers: authHeaders }
      );

      setActiveMessages((previousMessages) => [...previousMessages, response.data]);
      setConversationSummaries((previousSummaries) => ({
        ...previousSummaries,
        [selectedFriendId]: {
          ...(previousSummaries[selectedFriendId] || {}),
          last_message: getDirectMessagePreview(response.data),
          last_message_at: response.data?.created_at || new Date().toISOString(),
          unread_count: 0,
        },
      }));
    } catch (sendError) {
      console.error("Error sending room invite:", sendError);
      setError(sendError.response?.data?.detail || "Oda daveti gönderilirken bir hata oluştu.");
      await loadActiveRoomInviteTarget();
    } finally {
      setSendingInvite(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="messages-page">
      <div className="mx-auto max-w-6xl space-y-6" data-testid="messages-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="messages-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="messages-header-content">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-slate-900 " data-testid="messages-title">DM Kutum</h1>
                <p className="mt-2 text-base text-slate-600 " data-testid="messages-subtitle">
                  Arkadaşlarınla bire bir konuşmaları buradan başlatabilir ve sürdürebilirsin.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="messages-header-actions">
                <Button variant="outline" onClick={() => navigate("/friends")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="messages-go-friends-button">
                  <Users className="mr-2 h-4 w-4" /> Arkadaşlar
                </Button>
                <Button variant="outline" onClick={() => navigate("/dashboard")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="messages-go-dashboard-button">
                  <Home className="mr-2 h-4 w-4" /> Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700   " data-testid="messages-error-state">
            {error}
          </div>
        )}

        <Card className="overflow-hidden rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="messages-main-card">
          <div className="grid min-h-[72vh] grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className="border-b border-border/60 lg:border-b-0 lg:border-r" data-testid="messages-sidebar">
              <div className="border-b border-border/60 px-5 py-4">
                <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-muted-foreground" data-testid="messages-sidebar-title">
                  Arkadaşların
                </h2>
              </div>

              <ScrollArea className="h-[260px] lg:h-[72vh]" data-testid="messages-friends-scroll-area">
                <div className="space-y-2 p-3" data-testid="messages-friends-list">
                  {friendsLoading ? (
                    <p className="rounded-2xl px-3 py-8 text-center text-sm text-slate-600 " data-testid="messages-friends-loading">
                      Arkadaşlar yükleniyor...
                    </p>
                  ) : friends.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-border/70 bg-background/60 px-4 py-8 text-center" data-testid="messages-friends-empty-state">
                      <p className="text-sm font-medium text-slate-700 ">Henüz arkadaşın bulunmuyor.</p>
                      <p className="mt-1 text-xs text-slate-500 ">Mesajlaşmak için önce arkadaş eklemelisin.</p>
                    </div>
                  ) : (
                    friends.map((friend) => {
                      const isSelected = selectedFriendId === friend.profile_id;
                      const conversationSummary = conversationSummaries[friend.profile_id] || {};
                      const unreadCount = Number(conversationSummary.unread_count || 0);
                      const unreadLabel = unreadCount > 9 ? "9+" : unreadCount;
                      const lastMessagePreview = conversationSummary.last_message || "Henüz mesaj yok";
                      const lastMessageTime = formatSidebarTime(conversationSummary.last_message_at);

                      return (
                        <button
                          key={friend.profile_id}
                          type="button"
                          onClick={() => setSelectedFriendId(friend.profile_id)}
                          className={`flex w-full items-start gap-3 rounded-2xl border px-3.5 py-3 text-left transition-colors ${isSelected ? "border-primary/30 bg-primary/10" : "border-border/60 bg-background/40 hover:bg-background/70"}`}
                          data-testid={`messages-friend-item-${friend.profile_id}`}
                        >
                          <div className="flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden rounded-2xl bg-slate-900 text-sm font-bold text-white" data-testid={`messages-friend-avatar-${friend.profile_id}`}>
                            {friend.avatar_url ? (
                              <img src={friend.avatar_url} alt={`${getPublicUsername(friend)} avatar`} className="h-full w-full object-cover" data-testid={`messages-friend-avatar-image-${friend.profile_id}`} />
                            ) : (
                              getAvatarFallback(friend)
                            )}
                          </div>

                          <div className="min-w-0 flex-1">
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 flex-1">
                                <p className="truncate font-semibold text-slate-900 " data-testid={`messages-friend-username-${friend.profile_id}`}>
                                  {getPublicUsername(friend)}
                                </p>
                                <p className="mt-1 truncate text-sm text-slate-500 " data-testid={`messages-friend-preview-${friend.profile_id}`}>
                                  {lastMessagePreview}
                                </p>
                              </div>
                              <div className="flex shrink-0 flex-col items-end gap-1 pt-0.5">
                                {lastMessageTime && (
                                  <span className="text-[11px] font-medium text-slate-400 " data-testid={`messages-friend-time-${friend.profile_id}`}>
                                    {lastMessageTime}
                                  </span>
                                )}
                                {unreadCount > 0 && (
                                  <span className="inline-flex min-w-[20px] items-center justify-center rounded-full bg-red-500 px-1.5 py-0.5 text-[10px] font-semibold leading-none text-white shadow-sm" data-testid={`messages-friend-unread-badge-${friend.profile_id}`}>
                                    {unreadLabel}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })
                  )}
                </div>
              </ScrollArea>
            </div>

            <div className="flex min-h-[420px] flex-col" data-testid="messages-conversation-panel">
              {selectedFriend ? (
                <>
                  <div className="border-b border-border/60 px-5 py-4" data-testid="messages-conversation-header">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-lg font-semibold text-slate-900 " data-testid="messages-conversation-title">
                          {getPublicUsername(selectedFriend)}
                        </p>
                        {selectedFriendHandle && (
                          <p className="mt-1 text-sm text-slate-500 " data-testid="messages-conversation-handle">
                            {selectedFriendHandle}
                          </p>
                        )}
                      </div>

                      <div className="flex flex-col items-stretch gap-1.5 sm:items-end">
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={handleSendRoomInvite}
                          disabled={!activeRoomInviteTarget?.room_id || sendingInvite || roomInviteLoading}
                          className="rounded-xl border-border/70 bg-background hover:bg-secondary"
                          data-testid="messages-invite-room-button"
                        >
                          <Users className="h-4 w-4" /> {sendingInvite ? "Gönderiliyor..." : "Odaya Davet Et"}
                        </Button>
                        <p className="text-[11px] text-slate-500 " data-testid="messages-active-room-hint">
                          {roomInviteLoading
                            ? "Aktif room kontrol ediliyor..."
                            : activeRoomInviteTarget?.room_name
                              ? `Aktif room: ${activeRoomInviteTarget.room_name}`
                              : "Davet göndermek için önce aktif bir room içinde olmalısın."}
                        </p>
                      </div>
                    </div>
                  </div>

                  <ScrollArea className="flex-1 bg-background/30 px-5 py-5" data-testid="messages-conversation-scroll-area">
                    {messagesLoading ? (
                      <div className="flex h-full min-h-[280px] items-center justify-center" data-testid="messages-conversation-loading">
                        <p className="text-sm text-slate-600 ">Mesajlar yükleniyor...</p>
                      </div>
                    ) : activeMessages.length === 0 ? (
                      <div className="flex h-full min-h-[280px] items-center justify-center rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 text-center" data-testid="messages-conversation-empty-state">
                        <div>
                          <p className="text-base font-semibold text-slate-900 ">Henüz mesaj yok</p>
                          <p className="mt-2 text-sm text-slate-600 ">İlk mesajı göndererek konuşmayı başlat.</p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3" data-testid="messages-conversation-list">
                        {activeMessages.map((message) => {
                          const isOwnMessage = message.sender_uid === currentUser?.uid;
                          const isRoomInvite = message.message_type === ROOM_INVITE_MESSAGE_TYPE && message.room_invite?.room_id;

                          return (
                            <div key={message.id} className={`flex ${isOwnMessage ? "justify-end" : "justify-start"}`} data-testid={`messages-conversation-item-${message.id}`}>
                              {isRoomInvite ? (
                                <div className={`max-w-[92%] rounded-2xl border px-4 py-3 text-sm shadow-sm ${isOwnMessage ? "border-primary/25 bg-primary/10 text-foreground" : "border-border/70 bg-background text-foreground"}`} data-testid={`messages-room-invite-card-${message.id}`}>
                                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary/80">Oda Daveti</p>
                                  <p className="mt-2 leading-6 text-slate-700 ">{getRoomInviteCopy(message, isOwnMessage)}</p>
                                  {message.room_invite?.room_name && (
                                    <p className="mt-2 text-sm font-semibold text-slate-900 ">{message.room_invite.room_name}</p>
                                  )}
                                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                                    <p className="text-[11px] text-muted-foreground">{formatMessageTime(message.created_at)}</p>
                                    <Button
                                      type="button"
                                      size="sm"
                                      variant={isOwnMessage ? "secondary" : "outline"}
                                      className="h-8 rounded-lg px-3.5"
                                      onClick={() => navigate(`/room/${message.room_invite.room_id}`)}
                                      data-testid={`messages-room-invite-join-${message.id}`}
                                    >
                                      Katıl
                                    </Button>
                                  </div>
                                </div>
                              ) : (
                                <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm ${isOwnMessage ? "rounded-br-md bg-primary text-primary-foreground" : "rounded-bl-md border border-border/70 bg-background text-foreground"}`}>
                                  <p>{message.message}</p>
                                  <p className={`mt-2 text-[11px] ${isOwnMessage ? "text-primary-foreground/75" : "text-muted-foreground"}`}>{formatMessageTime(message.created_at)}</p>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </ScrollArea>

                  <form onSubmit={handleSendMessage} className="border-t border-border/60 bg-background/60 px-5 py-4" data-testid="messages-composer-form">
                    <div className="flex flex-col gap-3 sm:flex-row">
                      <Input
                        value={draftMessage}
                        onChange={(event) => setDraftMessage(event.target.value)}
                        placeholder="Mesajını yaz..."
                        className="h-11 rounded-xl"
                        data-testid="messages-composer-input"
                      />
                      <Button type="submit" className="h-11 rounded-xl px-5" disabled={!draftMessage.trim() || sendingMessage} data-testid="messages-composer-send-button">
                        <Send className="mr-2 h-4 w-4" /> {sendingMessage ? "Gönderiliyor..." : "Gönder"}
                      </Button>
                    </div>
                  </form>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center px-6 py-10" data-testid="messages-no-selection-state">
                  <div className="max-w-sm rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-10 text-center">
                    <MessageSquare className="mx-auto h-10 w-10 text-slate-400 " />
                    <p className="mt-4 text-base font-semibold text-slate-900 ">Mesajlaşmak için soldan bir arkadaş seç</p>
                    <p className="mt-2 text-sm text-slate-600 ">
                      Bir kişiyi seçtiğinde konuşma alanı ve mesaj yazma bölümü burada açılacak.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
