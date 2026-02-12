import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import axios from "axios";
import { API } from "@/App";
import { Home, Plus, LogIn } from "lucide-react";
import { saveRoom } from "@/lib/storage";

export default function Rooms() {
  const navigate = useNavigate();
  const userId = localStorage.getItem("userId");
  const [showProfile, setShowProfile] = useState(false);
  const [profileData, setProfileData] = useState({
    name: localStorage.getItem("userName") || "",
    study_field: ""
  });

  // Create Room
  const [createForm, setCreateForm] = useState({
    room_name: "",
    owner_name: profileData.name,
    owner_study_field: ""
  });

  // Join Room
  const [joinForm, setJoinForm] = useState({
    room_code: "",
    user_name: profileData.name,
    user_study_field: ""
  });

  const [loading, setLoading] = useState(false);

  const handleCreateRoom = async () => {
    if (!createForm.room_name.trim()) {
      alert("Oda adı boş olamaz");
      return;
    }
    if (!createForm.owner_name.trim()) {
      alert("İsim boş olamaz");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post(`${API}/rooms`, {
        name: createForm.room_name,
        owner_name: createForm.owner_name,
        owner_study_field: createForm.owner_study_field || null
      });

      const roomId = res.data.id;
      const userId = res.data.owner_id;

      // Store user info
      localStorage.setItem("currentRoomId", roomId);
      localStorage.setItem("currentUserId", userId);
      localStorage.setItem("userName", createForm.owner_name);
      
      // Save to localStorage persistence
      saveRoom(roomId, userId);

      navigate(`/room/${roomId}`);
    } catch (error) {
      console.error("Error creating room:", error);
      alert("Oda oluşturulurken hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const handleJoinRoom = async () => {
    if (!joinForm.room_code.trim()) {
      alert("Oda kodu boş olamaz");
      return;
    }
    if (!joinForm.user_name.trim()) {
      alert("İsim boş olamaz");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post(`${API}/rooms/join`, {
        room_code: joinForm.room_code,
        user_name: joinForm.user_name,
        user_study_field: joinForm.user_study_field || null
      });

      if (res.data.error) {
        alert(res.data.error);
        return;
      }

      const roomId = res.data.id;
      // Get the newly joined user (last in participants array)
      const userId = res.data.participants[res.data.participants.length - 1].id;

      // Store user info
      localStorage.setItem("currentRoomId", roomId);
      localStorage.setItem("currentUserId", userId);
      localStorage.setItem("userName", joinForm.user_name);
      
      // Save to localStorage persistence
      saveRoom(roomId, userId);

      navigate(`/room/${roomId}`);
    } catch (error) {
      console.error("Error joining room:", error);
      alert("Odaya katılırken hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const saveProfile = () => {
    if (!profileData.name.trim()) {
      alert("İsim boş olamaz");
      return;
    }
    localStorage.setItem("userName", profileData.name);
    setCreateForm({ ...createForm, owner_name: profileData.name, owner_study_field: profileData.study_field });
    setJoinForm({ ...joinForm, user_name: profileData.name, user_study_field: profileData.study_field });
    setShowProfile(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl font-extrabold text-gray-900 mb-3" data-testid="rooms-title">
            Online Çalışma Odaları
          </h1>
          <p className="text-lg text-gray-600">Arkadaşlarınla birlikte çalış, motivasyonunu artır</p>
          
          <div className="flex justify-center gap-3 mt-6">
            <Button variant="outline" onClick={() => navigate("/")} className="shadow-sm hover:shadow-md transition-shadow" data-testid="btn-home">
              <Home className="mr-2 h-4 w-4" />
              Ana Sayfa
            </Button>
            <Button variant="outline" onClick={() => setShowProfile(true)} className="shadow-sm hover:shadow-md transition-shadow" data-testid="btn-profile">
              👤 Profil
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <Card className="shadow-lg border border-gray-200">
          <CardContent className="pt-6">
            <Tabs defaultValue="create" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="create" data-testid="tab-create">
                  <Plus className="mr-2 h-4 w-4" />
                  Oda Oluştur
                </TabsTrigger>
                <TabsTrigger value="join" data-testid="tab-join">
                  <LogIn className="mr-2 h-4 w-4" />
                  Odaya Katıl
                </TabsTrigger>
              </TabsList>

              {/* Create Room Tab */}
              <TabsContent value="create" className="space-y-5 mt-6" data-testid="create-room-form">
                <div>
                  <Label htmlFor="room-name" className="font-semibold">Oda Adı *</Label>
                  <Input
                    id="room-name"
                    placeholder="Örn: TYT Matematik Çalışma"
                    value={createForm.room_name}
                    onChange={(e) => setCreateForm({ ...createForm, room_name: e.target.value })}
                    className="mt-2 border-2"
                    data-testid="input-room-name"
                  />
                </div>

                <div>
                  <Label htmlFor="owner-name" className="font-semibold">Adın *</Label>
                  <Input
                    id="owner-name"
                    placeholder="Örn: Ahmet"
                    value={createForm.owner_name}
                    onChange={(e) => setCreateForm({ ...createForm, owner_name: e.target.value })}
                    className="mt-2 border-2"
                    data-testid="input-owner-name"
                  />
                </div>

                <div>
                  <Label htmlFor="owner-field" className="font-semibold">Alan (Opsiyonel)</Label>
                  <select
                    id="owner-field"
                    className="w-full p-2 border-2 rounded mt-2"
                    value={createForm.owner_study_field}
                    onChange={(e) => setCreateForm({ ...createForm, owner_study_field: e.target.value })}
                    data-testid="select-owner-field"
                  >
                    <option value="">Seçiniz</option>
                    <option value="Sayısal">Sayısal</option>
                    <option value="EA">Eşit Ağırlık</option>
                    <option value="Sözel">Sözel</option>
                  </select>
                </div>

                <Button
                  className="w-full mt-2 shadow-md hover:shadow-lg transition-shadow font-semibold"
                  onClick={handleCreateRoom}
                  disabled={loading}
                  data-testid="btn-create-room"
                >
                  {loading ? "Oluşturuluyor..." : "Oda Oluştur"}
                </Button>
              </TabsContent>

              {/* Join Room Tab */}
              <TabsContent value="join" className="space-y-5 mt-6" data-testid="join-room-form">
                <div>
                  <Label htmlFor="room-code" className="font-semibold">Oda Kodu *</Label>
                  <Input
                    id="room-code"
                    placeholder="Örn: ABC123"
                    value={joinForm.room_code}
                    onChange={(e) => setJoinForm({ ...joinForm, room_code: e.target.value.toUpperCase() })}
                    className="mt-2 border-2"
                    data-testid="input-room-code"
                  />
                  <p className="text-sm text-gray-500 mt-2">Oda sahibinden kodu al</p>
                </div>

                <div>
                  <Label htmlFor="user-name" className="font-semibold">Adın *</Label>
                  <Input
                    id="user-name"
                    placeholder="Örn: Mehmet"
                    value={joinForm.user_name}
                    onChange={(e) => setJoinForm({ ...joinForm, user_name: e.target.value })}
                    className="mt-2 border-2"
                    data-testid="input-user-name"
                  />
                </div>

                <div>
                  <Label htmlFor="user-field" className="font-semibold">Alan (Opsiyonel)</Label>
                  <select
                    id="user-field"
                    className="w-full p-2 border-2 rounded mt-2"
                    value={joinForm.user_study_field}
                    onChange={(e) => setJoinForm({ ...joinForm, user_study_field: e.target.value })}
                    data-testid="select-user-field"
                  >
                    <option value="">Seçiniz</option>
                    <option value="Sayısal">Sayısal</option>
                    <option value="EA">Eşit Ağırlık</option>
                    <option value="Sözel">Sözel</option>
                  </select>
                </div>

                <Button
                  className="w-full mt-2 shadow-md hover:shadow-lg transition-shadow font-semibold"
                  onClick={handleJoinRoom}
                  disabled={loading}
                  data-testid="btn-join-room"
                >
                  {loading ? "Katılınıyor..." : "Odaya Katıl"}
                </Button>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Profile Dialog */}
      <Dialog open={showProfile} onOpenChange={setShowProfile}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Mini Profil</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="profile-name" className="font-semibold">Adın *</Label>
              <Input
                id="profile-name"
                value={profileData.name}
                onChange={(e) => setProfileData({ ...profileData, name: e.target.value })}
                className="mt-2 border-2"
                data-testid="input-profile-name"
              />
            </div>
            <div>
              <Label htmlFor="profile-field" className="font-semibold">Alan (Opsiyonel)</Label>
              <select
                id="profile-field"
                className="w-full p-2 border-2 rounded mt-2"
                value={profileData.study_field}
                onChange={(e) => setProfileData({ ...profileData, study_field: e.target.value })}
                data-testid="select-profile-field"
              >
                <option value="">Seçiniz</option>
                <option value="Sayısal">Sayısal</option>
                <option value="EA">Eşit Ağırlık</option>
                <option value="Sözel">Sözel</option>
              </select>
            </div>
            <Button onClick={saveProfile} className="w-full" data-testid="btn-save-profile">
              Kaydet
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
