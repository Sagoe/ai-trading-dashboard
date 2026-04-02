import { Routes, Route } from "react-router-dom";
import Dashboard    from "./pages/Dashboard";
import Markets      from "./pages/Markets";
import Predictions  from "./pages/Predictions";
import Portfolio    from "./pages/Portfolio";
import Settings     from "./pages/Settings";
import UploadPredict from "./pages/UploadPredict";

export default function App() {
  return (
    <Routes>
      <Route path="/"             element={<Dashboard />}     />
      <Route path="/markets"      element={<Markets />}       />
      <Route path="/predictions"  element={<Predictions />}   />
      <Route path="/portfolio"    element={<Portfolio />}     />
      <Route path="/upload"       element={<UploadPredict />} />
      <Route path="/settings"     element={<Settings />}      />
    </Routes>
  );
}
