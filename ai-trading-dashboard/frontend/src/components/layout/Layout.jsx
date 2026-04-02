import Sidebar    from "./Sidebar";
import Topbar     from "./Topbar";
import LiveTicker from "./LiveTicker";

export default function Layout({ children, onRefresh }) {
  return (
    <div className="min-h-screen bg-bg-primary">
      <Sidebar />
      {/* LiveTicker sits at the very top above Topbar */}
      <LiveTicker />
      {/* Topbar is pushed down by ticker height (h-10 = 40px) */}
      <div className="ml-60 pt-10">
        <Topbar onRefresh={onRefresh} />
        <main className="pt-14 min-h-screen">
          <div className="p-6 max-w-[1600px] mx-auto animate-fade-in-up">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
