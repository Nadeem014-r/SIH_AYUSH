import React, { useMemo, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ApiCamera, camerasApi, cameraStreamUrl } from '@/lib/api';
import { useSystemHealth } from '@/components/system/SystemHealthProvider';
import { CameraTile } from '@/components/live/CameraTile';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import {
  Grid2X2,
  Maximize2,
  Video,
  ArrowLeft,
  ShieldAlert,
  ShieldCheck,
  Shield,
  Plus,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';

export interface CameraItem extends Omit<ApiCamera, 'activityGate' | 'lowLightBoost'> {
  activityGate?: 'HIGH' | 'LOW';
  lowLightBoost?: boolean;
}

const LEGACY_CAMERAS_STORAGE_KEY = 'ibvap_cameras_data_v3';

export const LiveFeedsPage: React.FC = () => {
  const { cameras: apiCameras, health, reachable, refresh } = useSystemHealth();
  const cameras: CameraItem[] = useMemo(() => {
    const list =
      apiCameras.length > 0
        ? apiCameras
        : [
          {
            id: 'cam0',
            name: 'cam0',
            location: 'Border Area Camera 0',
            sector: 'North Border Sector',
            tier: 'red',
            isActive: true,
            fps: '30.0',
            activity: 'MOTION',
            health: 'online',
            source: 'pipeline',
          } as ApiCamera,
        ];
    return list.map((c) => ({
      ...c,
      streamUrl: cameraStreamUrl(c.id),
      activityGate: c.activityGate ?? undefined,
      lowLightBoost: c.lowLightBoost ?? undefined,
      tier: (c.tier as 'red' | 'yellow' | 'green') || 'red',
    }));
  }, [apiCameras]);
  const serverNow = health?.checkedAt ?? null;
  const [viewMode, setViewMode] = useState<'grid' | 'focus'>('grid');
  const [gridColumns, setGridColumns] = useState<'2' | '3'>('2');
  const [focusedCameraId, setFocusedCameraId] = useState<string>('cam0');
  const [selectedTierFilter, setSelectedTierFilter] = useState<'all' | 'red' | 'yellow' | 'green'>('all');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const [searchParams] = useSearchParams();
  const urlCamera = searchParams.get('camera');

  useEffect(() => {
    try {
      localStorage.removeItem(LEGACY_CAMERAS_STORAGE_KEY);
    } catch {
      // storage unavailable
    }
  }, []);

  useEffect(() => {
    if (urlCamera) {
      const match = cameras.find((c) => c.id.toLowerCase() === urlCamera.toLowerCase());
      if (match) {
        setFocusedCameraId(match.id);
        setViewMode('focus');
      }
    }
    if (searchParams.get('action') === 'add') {
      setIsAddModalOpen(true);
    }
  }, [urlCamera, searchParams, cameras]);

  // Form State for Adding a Camera
  const [newCamId, setNewCamId] = useState(`cam${cameras.length}`);
  const [newCamLocation, setNewCamLocation] = useState('');
  const [newCamSector, setNewCamSector] = useState('North Border Sector');
  const [newCamTier, setNewCamTier] = useState<'red' | 'yellow' | 'green'>('red');
  const [sourceType, setSourceType] = useState<'local' | 'rtsp'>('local');
  const [localCamIndex, setLocalCamIndex] = useState('0');
  const [newCamStreamUrl, setNewCamStreamUrl] = useState('');
  const [formError, setFormError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  const handleTileClick = (camId: string) => {
    setFocusedCameraId(camId);
    setViewMode('focus');
  };

  const handleRemoveCamera = async (camId: string) => {
    if (cameras.length <= 1) {
      showToast('Cannot remove last remaining surveillance feed');
      return;
    }
    try {
      const res = await camerasApi.deleteCamera(camId);
      if (res.isFallback) {
        showToast(`Could not remove ${camId.toUpperCase()}: ${res.error ?? 'backend unreachable'}`);
        return;
      }
      await refresh();
      if (focusedCameraId === camId) {
        setFocusedCameraId(cameras.find((c) => c.id !== camId)?.id || 'cam0');
      }
      showToast(`Removed camera channel ${camId.toUpperCase()}`);
    } catch (err) {
      showToast(
        `Failed to remove camera ${camId}: ${err instanceof Error ? err.message : 'unknown error'}`
      );
    }
  };

  const handleOpenAddModal = (defaultTier: 'red' | 'yellow' | 'green' = 'red') => {
    let nextNum = cameras.length;
    let candidateId = `cam${nextNum}`;
    while (cameras.some((c) => c.id.toLowerCase() === candidateId.toLowerCase())) {
      nextNum += 1;
      candidateId = `cam${nextNum}`;
    }
    setNewCamId(candidateId);
    setNewCamLocation('');
    setNewCamStreamUrl('');
    setNewCamTier(defaultTier);
    setFormError('');
    setIsSubmitting(false);
    setIsAddModalOpen(true);
  };

  const handleAddCameraSubmit = async (e?: React.FormEvent) => {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault();
    }
    if (isSubmitting) return;

    const cleanId = newCamId.trim().toLowerCase();
    const cleanLocation = newCamLocation.trim();

    if (!cleanId) {
      setFormError('Camera ID is required (e.g. cam1, cam_fence, etc.)');
      return;
    }

    if (!cleanLocation) {
      setFormError('Location or post name is required');
      return;
    }

    const finalStreamUrl = sourceType === 'local' ? localCamIndex : newCamStreamUrl.trim();

    const newCamera: CameraItem = {
      id: cleanId,
      name: cleanId,
      location: cleanLocation,
      sector: newCamSector,
      tier: newCamTier,
      streamUrl: finalStreamUrl || undefined,
      fps: '0.0',
      activity: '—',
      isActive: true,
      resolution: '1920x1080',
    };

    setIsSubmitting(true);
    setFormError('');
    try {
      const res = await camerasApi.addCamera(newCamera);
      if (res.isFallback) {
        setFormError(`Could not add the camera: ${res.error ?? 'backend unreachable'}`);
        setIsSubmitting(false);
        return;
      }
      await refresh();
      setIsSubmitting(false);
      setIsAddModalOpen(false);
      showToast(`Camera ${cleanId.toUpperCase()} (${newCamTier.toUpperCase()} ZONE) added successfully`);
    } catch (err) {
      setIsSubmitting(false);
      setFormError(
        `Failed to integrate camera with backend: ${err instanceof Error ? err.message : 'unknown error'
        }`
      );
    }
  };

  const focusedCamera = cameras.find((c) => c.id === focusedCameraId) || cameras[0];
  const onlineCount = cameras.filter((c) => c.health === 'online' && c.source !== 'idle').length;

  const redCameras = useMemo(() => cameras.filter((c) => (c.tier || 'red') === 'red'), [cameras]);
  const yellowCameras = useMemo(() => cameras.filter((c) => c.tier === 'yellow'), [cameras]);
  const greenCameras = useMemo(() => cameras.filter((c) => c.tier === 'green'), [cameras]);

  const renderCameraGrid = (cameraList: CameraItem[]) => (
    <div
      className={`grid gap-4 ${gridColumns === '3'
          ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
          : 'grid-cols-1 md:grid-cols-2'
        }`}
    >
      {cameraList.map((camera) => (
        <div
          key={camera.id}
          onClick={() => handleTileClick(camera.id)}
          className="cursor-pointer group/card focus:outline-none"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              handleTileClick(camera.id);
            }
          }}
        >
          <CameraTile
            cameraName={camera.name}
            label={camera.location}
            streamUrl={camera.streamUrl}
            isActive={camera.isActive}
            fps={camera.fps}
            activity={camera.activity}
            activityGate={camera.activityGate}
            lowLightBoost={camera.lowLightBoost}
            source={camera.source}
            health={camera.health}
            zones={camera.zones}
            detections={camera.detections}
            maxTier={camera.maxTier}
            tier={camera.tier}
            resolution={camera.resolution}
            lastFrameAt={camera.lastFrameAt}
            serverNow={serverNow}
            onToggleFocus={() => handleTileClick(camera.id)}
            onRemove={() => handleRemoveCamera(camera.id)}
          />
        </div>
      ))}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Dynamic Toast Feedback */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-2.5 bg-bg-surface border border-accent-teal/50 rounded-sm shadow-xl font-mono text-xs text-text-primary animate-in fade-in slide-in-from-bottom-2">
          <CheckCircle2 className="w-4 h-4 text-accent-teal" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Top Toolbar / Filter Row */}
      <div className="card-3d flex flex-col lg:flex-row lg:items-center justify-between gap-3 p-4 rounded-2xl border border-white/10 bg-gradient-to-b from-[#0c0c14] to-[#06060a] shadow-[0_15px_35px_rgba(0,0,0,0.8)]">
        {/* Left: Camera Count Indicator & Status */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Video className="w-4 h-4 text-accent-teal" />
            <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">
              Surveillance Grid
            </span>
          </div>

          <span className="text-white/20">|</span>

          <Badge
            variant={cameras.length > 0 && onlineCount === cameras.length ? 'green' : 'yellow'}
            dot
            size="sm"
          >
            {cameras.length} CAMERAS · {onlineCount} LIVE
          </Badge>

          {/* 3 Zone Filter Tabs */}
          <div className="flex items-center gap-1.5 p-1 bg-black/60 border border-white/10 rounded-xl">
            <button
              onClick={() => setSelectedTierFilter('all')}
              className={`px-2.5 py-1 text-xs font-mono font-medium rounded-lg transition-all ${selectedTierFilter === 'all'
                  ? 'bg-white/15 text-white shadow-sm'
                  : 'text-text-dim hover:text-white'
                }`}
            >
              ALL ({cameras.length})
            </button>
            <button
              onClick={() => setSelectedTierFilter('red')}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono font-medium rounded-lg transition-all ${selectedTierFilter === 'red'
                  ? 'bg-accent-red/25 text-accent-red border border-accent-red/40 shadow-sm'
                  : 'text-text-dim hover:text-accent-red'
                }`}
            >
              <span className="w-2 h-2 rounded-full bg-accent-red animate-pulse" />
              RED ({redCameras.length})
            </button>
            <button
              onClick={() => setSelectedTierFilter('yellow')}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono font-medium rounded-lg transition-all ${selectedTierFilter === 'yellow'
                  ? 'bg-accent-yellow/25 text-accent-yellow border border-accent-yellow/40 shadow-sm'
                  : 'text-text-dim hover:text-accent-yellow'
                }`}
            >
              <span className="w-2 h-2 rounded-full bg-accent-yellow" />
              YELLOW ({yellowCameras.length})
            </button>
            <button
              onClick={() => setSelectedTierFilter('green')}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono font-medium rounded-lg transition-all ${selectedTierFilter === 'green'
                  ? 'bg-accent-green/25 text-accent-green border border-accent-green/40 shadow-sm'
                  : 'text-text-dim hover:text-accent-green'
                }`}
            >
              <span className="w-2 h-2 rounded-full bg-accent-green" />
              GREEN ({greenCameras.length})
            </button>
          </div>
        </div>

        {/* Right: Actions & Layout Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus className="w-4 h-4" />}
            onClick={() => handleOpenAddModal('red')}
          >
            Add Camera
          </Button>

          {viewMode === 'focus' && (
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<ArrowLeft className="w-3.5 h-3.5" />}
              onClick={() => setViewMode('grid')}
            >
              Back to Grid
            </Button>
          )}

          {viewMode === 'grid' && cameras.length >= 4 && (
            <div className="hidden sm:inline-flex p-1 bg-black/60 border border-white/10 rounded-xl shadow-inner">
              <button
                onClick={() => setGridColumns('2')}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition-all ${gridColumns === '2'
                    ? 'bg-accent-teal/15 text-accent-teal font-bold border border-accent-teal/40'
                    : 'text-text-dim hover:text-white'
                  }`}
              >
                2 COL
              </button>
              <button
                onClick={() => setGridColumns('3')}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition-all ${gridColumns === '3'
                    ? 'bg-accent-teal/15 text-accent-teal font-bold border border-accent-teal/40'
                    : 'text-text-dim hover:text-white'
                  }`}
              >
                3 COL
              </button>
            </div>
          )}

          <div className="inline-flex p-1 bg-black/60 border border-white/10 rounded-xl shadow-inner">
            <button
              onClick={() => setViewMode('grid')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-mono font-semibold rounded-lg transition-all ${viewMode === 'grid'
                  ? 'bg-accent-teal/15 text-accent-teal border border-accent-teal/40'
                  : 'text-text-dim hover:text-white'
                }`}
            >
              <Grid2X2 className="w-3.5 h-3.5" />
              <span>GRID</span>
            </button>

            <button
              onClick={() => setViewMode('focus')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-mono font-semibold rounded-lg transition-all ${viewMode === 'focus'
                  ? 'bg-accent-teal/15 text-accent-teal border border-accent-teal/40'
                  : 'text-text-dim hover:text-white'
                }`}
            >
              <Maximize2 className="w-3.5 h-3.5" />
              <span>FOCUS</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      {reachable === null && cameras.length === 0 ? (
        <div role="status" className="card-3d p-10 border border-white/10 rounded-2xl text-center text-xs font-mono text-text-dim">
          Loading cameras…
        </div>
      ) : reachable === false && cameras.length === 0 ? (
        <div role="alert" className="card-3d p-10 border border-accent-red/40 bg-accent-red/5 rounded-2xl text-center space-y-2">
          <AlertCircle className="w-6 h-6 text-accent-red mx-auto" />
          <h3 className="text-sm font-semibold text-white">Backend unreachable</h3>
          <p className="text-xs text-text-dim">Start it with ./run.sh up — this page reconnects automatically.</p>
          <Button variant="secondary" size="sm" onClick={() => refresh()}>
            Retry now
          </Button>
        </div>
      ) : cameras.length === 0 ? (
        <div className="card-3d p-10 border border-dashed border-white/15 rounded-2xl text-center space-y-3">
          <Video className="w-8 h-8 text-text-muted mx-auto" />
          <h3 className="text-base font-semibold text-white">No cameras configured</h3>
          <p className="text-xs text-text-dim max-w-md mx-auto">
            Choose a zone criticality level to add your first surveillance channel to the grid.
          </p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <Button variant="primary" size="sm" onClick={() => handleOpenAddModal('red')}>
              + Add Red Camera
            </Button>
            <Button variant="secondary" size="sm" onClick={() => handleOpenAddModal('yellow')}>
              + Add Yellow Camera
            </Button>
            <Button variant="secondary" size="sm" onClick={() => handleOpenAddModal('green')}>
              + Add Green Camera
            </Button>
          </div>
        </div>
      ) : viewMode === 'focus' ? (
        /* Focus View */
        <div className="space-y-4">
          <div className="flex items-center gap-2 overflow-x-auto pb-1.5">
            <span className="font-mono text-xs text-text-dim uppercase tracking-wider shrink-0 mr-1">
              Select Camera:
            </span>
            {cameras.map((cam) => (
              <button
                key={cam.id}
                onClick={() => setFocusedCameraId(cam.id)}
                className={`px-3 py-1 text-xs font-mono rounded-sm transition-all flex items-center gap-2 border shrink-0 ${focusedCameraId === cam.id
                    ? 'bg-accent-teal/20 text-accent-teal border-accent-teal/50 font-semibold shadow-sm'
                    : 'bg-bg-surface text-text-dim border-border-subtle hover:text-text-primary hover:bg-bg-elevated'
                  }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${cam.tier === 'red'
                      ? 'bg-accent-red'
                      : cam.tier === 'yellow'
                        ? 'bg-accent-yellow'
                        : 'bg-accent-green'
                    }`}
                />
                <span>{cam.name}</span>
                <span className="text-[10px] text-text-muted hidden sm:inline">
                  ({cam.location})
                </span>
              </button>
            ))}
          </div>

          <div className="max-w-5xl mx-auto">
            <CameraTile
              cameraName={focusedCamera.name}
              label={focusedCamera.location}
              streamUrl={focusedCamera.streamUrl}
              isActive={focusedCamera.isActive}
              fps={focusedCamera.fps}
              activity={focusedCamera.activity}
              activityGate={focusedCamera.activityGate}
              lowLightBoost={focusedCamera.lowLightBoost}
              source={focusedCamera.source}
              health={focusedCamera.health}
              zones={focusedCamera.zones}
              detections={focusedCamera.detections}
              maxTier={focusedCamera.maxTier}
              tier={focusedCamera.tier}
              resolution={focusedCamera.resolution}
              lastFrameAt={focusedCamera.lastFrameAt}
              serverNow={serverNow}
              isFocused={true}
              onToggleFocus={() => setViewMode('grid')}
              onRemove={() => handleRemoveCamera(focusedCamera.id)}
              className="shadow-2xl"
            />
          </div>
        </div>
      ) : (
        /* 3 SECTIONS SURVEILLANCE GRID */
        <div className="space-y-8">
          {/* SECTION 1: RED ZONE CAMERAS */}
          {(selectedTierFilter === 'all' || selectedTierFilter === 'red') && (
            <div className="space-y-3.5 p-4 rounded-2xl border border-accent-red/20 bg-accent-red/[0.02] shadow-[0_4px_25px_rgba(239,68,68,0.05)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-accent-red/15">
                <div className="flex items-center gap-2.5">
                  <div className="p-1.5 rounded-lg bg-accent-red/20 border border-accent-red/40 text-accent-red">
                    <ShieldAlert className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                        Red Critical Sector Cameras
                      </h3>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-accent-red/20 text-accent-red border border-accent-red/40">
                        {redCameras.length} CHANNELS
                      </span>
                    </div>
                    <p className="text-[11px] text-text-dim">
                      <strong className="text-accent-red font-medium">Detection Rules:</strong> Zero-tolerance perimeter breach · Immediate High Intrusion Alarms (Instant Threat Score 25–100).
                    </p>
                  </div>
                </div>

                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={<Plus className="w-3.5 h-3.5" />}
                  onClick={() => handleOpenAddModal('red')}
                  className="border-accent-red/40 text-accent-red hover:bg-accent-red/10"
                >
                  + Add Red Camera
                </Button>
              </div>

              {redCameras.length > 0 ? (
                renderCameraGrid(redCameras)
              ) : (
                <div className="p-6 text-center border border-dashed border-accent-red/20 rounded-xl text-xs font-mono text-text-dim">
                  No Red Critical cameras added. Click "+ Add Red Camera" to register high-risk border fence feeds.
                </div>
              )}
            </div>
          )}

          {/* SECTION 2: YELLOW ZONE CAMERAS */}
          {(selectedTierFilter === 'all' || selectedTierFilter === 'yellow') && (
            <div className="space-y-3.5 p-4 rounded-2xl border border-accent-yellow/20 bg-accent-yellow/[0.02] shadow-[0_4px_25px_rgba(234,179,8,0.05)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-accent-yellow/15">
                <div className="flex items-center gap-2.5">
                  <div className="p-1.5 rounded-lg bg-accent-yellow/20 border border-accent-yellow/40 text-accent-yellow">
                    <Shield className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                        Yellow Buffer Sector Cameras
                      </h3>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-accent-yellow/20 text-accent-yellow border border-accent-yellow/40">
                        {yellowCameras.length} CHANNELS
                      </span>
                    </div>
                    <p className="text-[11px] text-text-dim">
                      <strong className="text-accent-yellow font-medium">Detection Rules:</strong> Inward approach velocity tracking · Loiter dwell time (&gt;30s) · Group movement staging rules.
                    </p>
                  </div>
                </div>

                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={<Plus className="w-3.5 h-3.5" />}
                  onClick={() => handleOpenAddModal('yellow')}
                  className="border-accent-yellow/40 text-accent-yellow hover:bg-accent-yellow/10"
                >
                  + Add Yellow Camera
                </Button>
              </div>

              {yellowCameras.length > 0 ? (
                renderCameraGrid(yellowCameras)
              ) : (
                <div className="p-6 text-center border border-dashed border-accent-yellow/20 rounded-xl text-xs font-mono text-text-dim">
                  No Yellow Buffer cameras added. Click "+ Add Yellow Camera" to register approach corridor feeds.
                </div>
              )}
            </div>
          )}

          {/* SECTION 3: GREEN ZONE CAMERAS */}
          {(selectedTierFilter === 'all' || selectedTierFilter === 'green') && (
            <div className="space-y-3.5 p-4 rounded-2xl border border-accent-green/20 bg-accent-green/[0.02] shadow-[0_4px_25px_rgba(34,197,94,0.05)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-accent-green/15">
                <div className="flex items-center gap-2.5">
                  <div className="p-1.5 rounded-lg bg-accent-green/20 border border-accent-green/40 text-accent-green">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                        Green Base / Safe Sector Cameras
                      </h3>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-accent-green/20 text-accent-green border border-accent-green/40">
                        {greenCameras.length} CHANNELS
                      </span>
                    </div>
                    <p className="text-[11px] text-text-dim">
                      <strong className="text-accent-green font-medium">Detection Rules:</strong> Normal base camp activity allowed · Night curfew automatic shift alerts.
                    </p>
                  </div>
                </div>

                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={<Plus className="w-3.5 h-3.5" />}
                  onClick={() => handleOpenAddModal('green')}
                  className="border-accent-green/40 text-accent-green hover:bg-accent-green/10"
                >
                  + Add Green Camera
                </Button>
              </div>

              {greenCameras.length > 0 ? (
                renderCameraGrid(greenCameras)
              ) : (
                <div className="p-6 text-center border border-dashed border-accent-green/20 rounded-xl text-xs font-mono text-text-dim">
                  No Green Base cameras added. Click "+ Add Green Camera" to register interior outpost feeds.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Add Camera Modal Dialog */}
      <Modal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        title="Add Surveillance Camera Channel"
        description="Select a criticality zone and configure the camera channel rules."
        size="md"
        footer={
          <div className="flex items-center justify-between w-full">
            <span />
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsAddModalOpen(false)}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                disabled={isSubmitting}
                leftIcon={<Plus className="w-3.5 h-3.5" />}
                onClick={() => handleAddCameraSubmit()}
              >
                {isSubmitting ? 'Registering Camera…' : 'Register Camera'}
              </Button>
            </div>
          </div>
        }
      >
        <form onSubmit={handleAddCameraSubmit} className="space-y-4">
          {formError && (
            <div className="p-2.5 bg-accent-red/15 border border-accent-red/40 rounded-sm flex items-center gap-2 text-xs text-accent-red">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{formError}</span>
            </div>
          )}

          {/* 3 ZONE CRITICALITY SELECTOR */}
          <div>
            <label className="block text-xs font-mono uppercase text-text-dim mb-2">
              Select Zone Criticality & Detection Rules <span className="text-accent-red">*</span>
            </label>
            <div className="grid grid-cols-3 gap-2.5">
              {/* Option 1: Red */}
              <button
                type="button"
                onClick={() => setNewCamTier('red')}
                className={`p-3 rounded-xl border text-left transition-all ${newCamTier === 'red'
                    ? 'bg-accent-red/20 border-accent-red shadow-[0_0_15px_rgba(239,68,68,0.25)] ring-1 ring-accent-red'
                    : 'bg-bg-elevated border-white/10 hover:border-accent-red/40'
                  }`}
              >
                <div className="flex items-center gap-1.5 text-accent-red font-bold font-mono text-xs mb-1">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  <span>RED ZONE</span>
                </div>
                <div className="text-[10px] text-text-dim leading-tight">
                  Critical Fence · Zero-tolerance breach rules
                </div>
              </button>

              {/* Option 2: Yellow */}
              <button
                type="button"
                onClick={() => setNewCamTier('yellow')}
                className={`p-3 rounded-xl border text-left transition-all ${newCamTier === 'yellow'
                    ? 'bg-accent-yellow/20 border-accent-yellow shadow-[0_0_15px_rgba(234,179,8,0.25)] ring-1 ring-accent-yellow'
                    : 'bg-bg-elevated border-white/10 hover:border-accent-yellow/40'
                  }`}
              >
                <div className="flex items-center gap-1.5 text-accent-yellow font-bold font-mono text-xs mb-1">
                  <Shield className="w-3.5 h-3.5" />
                  <span>YELLOW ZONE</span>
                </div>
                <div className="text-[10px] text-text-dim leading-tight">
                  Buffer Corridor · Approach & loitering rules
                </div>
              </button>

              {/* Option 3: Green */}
              <button
                type="button"
                onClick={() => setNewCamTier('green')}
                className={`p-3 rounded-xl border text-left transition-all ${newCamTier === 'green'
                    ? 'bg-accent-green/20 border-accent-green shadow-[0_0_15px_rgba(34,197,94,0.25)] ring-1 ring-accent-green'
                    : 'bg-bg-elevated border-white/10 hover:border-accent-green/40'
                  }`}
              >
                <div className="flex items-center gap-1.5 text-accent-green font-bold font-mono text-xs mb-1">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>GREEN ZONE</span>
                </div>
                <div className="text-[10px] text-text-dim leading-tight">
                  Base Camp · Safe area & night curfew rules
                </div>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-mono uppercase text-text-dim mb-1">
                Camera ID <span className="text-accent-red">*</span>
              </label>
              <input
                type="text"
                value={newCamId}
                onChange={(e) => setNewCamId(e.target.value)}
                placeholder="e.g. cam4"
                className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal font-mono"
              />
              <span className="text-[10px] text-text-muted mt-0.5 block font-mono">
                Identifier used in AI pipelines
              </span>
            </div>

            <div>
              <label className="block text-xs font-mono uppercase text-text-dim mb-1">
                Sector / Area
              </label>
              <select
                value={newCamSector}
                onChange={(e) => setNewCamSector(e.target.value)}
                className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal"
              >
                <option value="North Border Sector">North Border Sector</option>
                <option value="East Border Sector">East Border Sector</option>
                <option value="South Perimeter">South Perimeter</option>
                <option value="West Mountain Sector">West Mountain Sector</option>
                <option value="Checkpost Bravo Corridor">Checkpost Corridor</option>
                <option value="Riverine Border Zone">Riverine Border Zone</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-mono uppercase text-text-dim mb-1">
              Camera Location / Post Name <span className="text-accent-red">*</span>
            </label>
            <input
              type="text"
              value={newCamLocation}
              onChange={(e) => setNewCamLocation(e.target.value)}
              placeholder="e.g. Outpost Delta Watchtower, Gate 3 Checkpoint"
              className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-mono uppercase text-text-dim mb-1">
                Source Type
              </label>
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value as 'local' | 'rtsp')}
                className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal"
              >
                <option value="local">Local USB/Front Camera</option>
                <option value="rtsp">RTSP / IP Stream</option>
              </select>
            </div>

            {sourceType === 'local' ? (
              <div>
                <label className="block text-xs font-mono uppercase text-text-dim mb-1">
                  Device Index
                </label>
                <select
                  value={localCamIndex}
                  onChange={(e) => setLocalCamIndex(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal font-mono"
                >
                  <option value="0">Camera 0 (Default Front)</option>
                  <option value="1">Camera 1 (External)</option>
                  <option value="2">Camera 2</option>
                  <option value="3">Camera 3</option>
                </select>
              </div>
            ) : (
              <div>
                <label className="block text-xs font-mono uppercase text-text-dim mb-1">
                  RTSP URL
                </label>
                <input
                  type="text"
                  value={newCamStreamUrl}
                  onChange={(e) => setNewCamStreamUrl(e.target.value)}
                  placeholder="rtsp://192.168.1.104:554/h264/ch1/main"
                  className="w-full px-3 py-2 bg-bg-elevated border border-border-subtle rounded-sm text-sm text-text-primary focus:outline-none focus:border-accent-teal font-mono text-xs"
                />
              </div>
            )}
            <span className="text-[10px] text-text-muted mt-0.5 block col-span-2">
              {sourceType === 'local'
                ? 'Select the hardware device index of the local camera.'
                : 'Enter the RTSP link for the IP camera.'}
            </span>
          </div>
        </form>
      </Modal>
    </div>
  );
};
