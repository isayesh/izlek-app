import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Home, MessageSquare, Send, Users } from "lucide-react";

import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAuth } from "@/contexts/AuthContext";
import { API } from "@/App";
import { formatPublicHandle, getAvatarFallback, getPublicUsername } from "@/lib/publicProfile";

const formatMessageTime = (timestamp) => new Date(timestamp).toLocaleTimeString("tr-TR", {
  hour: "2-digit",
  minute: "2-digit",
});

export default function Messages() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [friends, setFriends] = useState([]);
  const [friendsLoading, setFriendsLoading] = useState(true);
  const [selectedFriendId, setSelectedFriendId] = useState("");
  const [draftMessage, setDraftMessage] = useState("");
  const [conversations, setConversations] = useState({});
  const [error, setError] = useState("");

  const authHeaders = useMemo(() => (
    currentUser?.uid ? { "X-Firebase-UID": currentUser.uid } : {}
  ), [currentUser]);

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

  useEffect(() => {
    loadFriends();
  }, [loadFriends]);

  const selectedFriend = useMemo(
    () => friends.find((friend) => friend.profile_id === selectedFriendId) || null,
    [friends, selectedFriendId]
  );

  const activeMessages = selectedFriendId ? conversations[selectedFriendId] || [] : [];

  const handleSendMessage = (event) => {
    event.preventDefault();

    const trimmedMessage = draftMessage.trim();
    if (!selectedFriendId || !trimmedMessage) {
      return;
    }

    const newMessage = {
      id: `${selectedFriendId}-${Date.now()}`,
      content: trimmedMessage,
      timestamp: new Date().toISOString(),
      sender: "me",
    };

    setConversations((previousConversations) => ({
      ...previousConversations,
      [selectedFriendId]: [...(previousConversations[selectedFriendId] || []), newMessage],
    }));
    setDraftMessage("");
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="messages-page">
      <div className="mx-auto max-w-6xl space-y-6" data-testid="messages-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="messages-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="messages-header-content">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="messages-title">DM Kutum</h1>
                <p className="mt-2 text-base text-slate-600 dark:text-slate-300" data-testid="messages-subtitle">
                  Arkadaşlarınla bire bir konuşmaları buradan başlatabilir ve sürdürebilirsin.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="messages-header-actions">
                <ThemeToggle className="h-10 w-10 rounded-xl border border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100" />
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
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300" data-testid="messages-error-state">
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
                    <p className="rounded-2xl px-3 py-8 text-center text-sm text-slate-600 dark:text-slate-300" data-testid="messages-friends-loading">
                      Arkadaşlar yükleniyor...
                    </p>
                  ) : friends.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-border/70 bg-background/60 px-4 py-8 text-center" data-testid="messages-friends-empty-state">
                      <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Henüz arkadaşın bulunmuyor.</p>
                      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Mesajlaşmak için önce arkadaş eklemelisin.</p>
                    </div>
                  ) : (
                    friends.map((friend) => {
                      const handleText = formatPublicHandle(friend.handle_display || friend.handle);
                      const isSelected = selectedFriendId === friend.profile_id;

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
                            <p className="truncate font-semibold text-slate-900 dark:text-slate-100" data-testid={`messages-friend-username-${friend.profile_id}`}>
                              {getPublicUsername(friend)}
                            </p>
                            {handleText && (
                              <p className="mt-1 truncate text-sm text-slate-500 dark:text-slate-400" data-testid={`messages-friend-handle-${friend.profile_id}`}>
                                {handleText}
                              </p>
                            )}
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
                    <p className="text-lg font-semibold text-slate-900 dark:text-slate-100" data-testid="messages-conversation-title">
                      {getPublicUsername(selectedFriend)}
                    </p>
                    {selectedFriend.handle && (
                      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400" data-testid="messages-conversation-handle">
                        {formatPublicHandle(selectedFriend.handle_display || selectedFriend.handle)}
                      </p>
                    )}
                  </div>

                  <ScrollArea className="flex-1 bg-background/30 px-5 py-5" data-testid="messages-conversation-scroll-area">
                    {activeMessages.length === 0 ? (
                      <div className="flex h-full min-h-[280px] items-center justify-center rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 text-center" data-testid="messages-conversation-empty-state">
                        <div>
                          <p className="text-base font-semibold text-slate-900 dark:text-slate-100">Henüz mesaj yok</p>
                          <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">İlk mesajı göndererek konuşmayı başlat.</p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3" data-testid="messages-conversation-list">
                        {activeMessages.map((message) => (
                          <div key={message.id} className="flex justify-end" data-testid={`messages-conversation-item-${message.id}`}>
                            <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-3 text-sm text-primary-foreground shadow-sm">
                              <p>{message.content}</p>
                              <p className="mt-2 text-[11px] text-primary-foreground/75">{formatMessageTime(message.timestamp)}</p>
                            </div>
                          </div>
                        ))}
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
                      <Button type="submit" className="h-11 rounded-xl px-5" disabled={!draftMessage.trim()} data-testid="messages-composer-send-button">
                        <Send className="mr-2 h-4 w-4" /> Gönder
                      </Button>
                    </div>
                  </form>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center px-6 py-10" data-testid="messages-no-selection-state">
                  <div className="max-w-sm rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-10 text-center">
                    <MessageSquare className="mx-auto h-10 w-10 text-slate-400 dark:text-slate-500" />
                    <p className="mt-4 text-base font-semibold text-slate-900 dark:text-slate-100">Mesajlaşmak için soldan bir arkadaş seç</p>
                    <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
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
