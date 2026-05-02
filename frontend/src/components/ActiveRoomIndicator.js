import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

const parseActiveRoomId = () => {
  const activeRoomRaw = localStorage.getItem("active_room");

  if (activeRoomRaw) {
    try {
      const parsed = JSON.parse(activeRoomRaw);
      if (typeof parsed === "string" && parsed.trim()) {
        return parsed;
      }

      if (parsed && typeof parsed === "object") {
        return parsed.roomId || parsed.room_id || parsed.id || "";
      }
    } catch {
      if (activeRoomRaw.trim()) {
        return activeRoomRaw;
      }
    }
  }

  const currentRoomId = localStorage.getItem("currentRoomId");
  return currentRoomId?.trim() || "";
};

export default function ActiveRoomIndicator() {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeRoomId, setActiveRoomId] = useState("");

  useEffect(() => {
    setActiveRoomId(parseActiveRoomId());
  }, [location.pathname]);

  const isRoomPage = useMemo(
    () => location.pathname.startsWith("/room/"),
    [location.pathname]
  );

  const isAllowedPage = useMemo(
    () => [
      "/dashboard",
      "/messages",
      "/program/create",
      "/profile",
      "/profile/edit",
      "/friends",
      "/notifications",
      "/leaderboard",
      "/net-tracking",
      "/rooms",
      "/forum",
    ].includes(location.pathname),
    [location.pathname]
  );

  if (!activeRoomId || isRoomPage || !isAllowedPage) {
    return null;
  }

  const handleGoBackToRoom = () => {
    navigate(`/room/${activeRoomId}`);
  };

  const handleLeaveActiveRoom = () => {
    localStorage.removeItem("active_room");
    localStorage.removeItem("currentRoomId");
    setActiveRoomId("");
  };

  return (
    <div className="pointer-events-none fixed left-1/2 top-3 z-50 w-[calc(100vw-1rem)] max-w-2xl -translate-x-1/2 sm:top-4 sm:w-[calc(100vw-2rem)]" data-testid="active-room-indicator-wrapper">
      <div className="pointer-events-auto mx-auto flex min-w-0 items-center justify-between gap-2 rounded-xl border border-indigo-100 bg-white/96 px-3 py-2 shadow-[0_14px_28px_-20px_rgba(99,102,241,0.7)] backdrop-blur" data-testid="active-room-indicator">
        <p className="min-w-0 truncate text-xs font-medium text-slate-700 sm:text-sm" data-testid="active-room-indicator-text">
          Şu anda bir odadasın
        </p>
        <div className="flex shrink-0 items-center gap-1.5" data-testid="active-room-indicator-actions">
          <Button
            type="button"
            size="sm"
            onClick={handleGoBackToRoom}
            className="h-8 rounded-lg bg-[#6366F1] px-2.5 text-xs font-medium text-white hover:bg-[#4F46E5] sm:px-3"
            data-testid="active-room-go-button"
          >
            Odaya Dön
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={handleLeaveActiveRoom}
            className="h-8 rounded-lg border-slate-200 px-2.5 text-xs font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-700 sm:px-3"
            data-testid="active-room-leave-button"
          >
            Ayrıl
          </Button>
        </div>
      </div>
    </div>
  );
}
