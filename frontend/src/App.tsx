import { useEffect, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { LoadingScreen } from "./components/LoadingScreen";
import Dashboard from "./pages/Dashboard";
import Live from "./pages/Live";
import Employees from "./pages/Employees";
import FaceRecognition from "./pages/FaceRecognition";
import Events from "./pages/Events";
import Models from "./pages/Models";
import Settings from "./pages/Settings";
import Attendance from "./pages/Attendance";
import ElectricalDataset from "./pages/ElectricalDataset";
import Training from "./pages/Training";
import TrainingProgress from "./pages/TrainingProgress";
import ModelComparison from "./pages/ModelComparison";
import ReferenceDesign from "./pages/ReferenceDesign";
import ReferencePanels from "./pages/ReferencePanels";
import TopologyViewer from "./pages/TopologyViewer";
import Datasheets from "./pages/Datasheets";
import PanelInspector from "./pages/PanelInspector";
import Inspection from "./pages/Inspection";
import ImageAnalysis from "./pages/ImageAnalysis";
import ImageComparison from "./pages/ImageComparison";
import Reports from "./pages/Reports";
import Metrics from "./pages/Metrics";
import Login from "./pages/Login";

function AppShell() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<Live />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/employees" element={<Employees />} />
        <Route path="/face-recognition" element={<FaceRecognition />} />
        <Route path="/attendance" element={<Attendance />} />
        <Route path="/events" element={<Events />} />
        <Route path="/datasets" element={<ElectricalDataset />} />
        <Route path="/training" element={<Training />} />
        <Route path="/training/:id" element={<TrainingProgress />} />
        <Route path="/training/:id/comparison" element={<ModelComparison />} />
        <Route path="/reference" element={<ReferenceDesign />} />
        <Route path="/reference-panels" element={<ReferencePanels />} />
        <Route path="/topology" element={<TopologyViewer />} />
        <Route path="/datasheets" element={<Datasheets />} />
        <Route path="/panel" element={<PanelInspector />} />
        <Route path="/inspector" element={<Navigate to="/panel" replace />} />
        <Route path="/inspection" element={<Inspection />} />
        <Route path="/image-analysis" element={<ImageAnalysis />} />
        <Route path="/image-comparison" element={<ImageComparison />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/models" element={<Models />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default function App() {
  // Brief branded splash on first load, then fade out.
  const [showSplash, setShowSplash] = useState(true);
  const [fading, setFading] = useState(false);
  useEffect(() => {
    const t1 = setTimeout(() => setFading(true), 850);
    const t2 = setTimeout(() => setShowSplash(false), 1350);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  return (
    <>
      {showSplash && <LoadingScreen fading={fading} />}
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={<AppShell />} />
      </Routes>
    </>
  );
}
