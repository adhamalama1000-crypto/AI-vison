import { useMemo, useState } from "react";
import {
  Plus, Search, ChevronLeft, ChevronRight, Pencil, Trash2, Users, ScanFace, ImageIcon,
} from "lucide-react";
import { useCameras, useEmployees, useInvalidate } from "../hooks/useData";
import { EmployeeDialog } from "../components/EmployeeDialog";
import { Card, Badge, EmptyState, Skeleton } from "../components/ui/primitives";
import { ConfirmDialog } from "../components/ui/Dialog";
import { api, mediaUrl } from "../lib/api";
import type { Employee } from "../lib/types";
import { fmt, timeAgo, cx } from "../lib/format";
import { useToast } from "../hooks/useToast";

const PER_PAGE = 8;

export default function Employees() {
  const { data, isLoading } = useEmployees();
  const { data: camData } = useCameras();
  const invalidate = useInvalidate();
  const { push } = useToast();

  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<Employee | null>(null);
  const [toDelete, setToDelete] = useState<Employee | null>(null);
  const [deleting, setDeleting] = useState(false);

  const employees = data?.employees ?? [];
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return employees;
    return employees.filter((e) =>
      e.full_name.toLowerCase().includes(q) ||
      (e.employee_code || "").toLowerCase().includes(q) ||
      (e.department || "").toLowerCase().includes(q) ||
      (e.job_title || "").toLowerCase().includes(q));
  }, [employees, search]);

  const pages = Math.max(1, Math.ceil(filtered.length / PER_PAGE));
  const clampedPage = Math.min(page, pages - 1);
  const shown = filtered.slice(clampedPage * PER_PAGE, clampedPage * PER_PAGE + PER_PAGE);

  const openAdd = () => { setEditing(null); setDialogOpen(true); };
  const openEdit = (e: Employee) => { setEditing(e); setDialogOpen(true); };

  const confirmDelete = async () => {
    if (!toDelete) return;
    setDeleting(true);
    try { await api.deleteEmployee(toDelete.id); push("Employee deleted", "success"); invalidate("employees", "dashboard"); }
    catch (e: any) { push(e.message, "error"); }
    finally { setDeleting(false); setToDelete(null); }
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative max-w-sm flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" />
          <input className="input pl-9" placeholder="Search employees…" value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }} />
        </div>
        <button className="btn-primary" onClick={openAdd}><Plus className="h-4 w-4" /> Add employee</button>
      </div>

      <Card className="overflow-hidden p-0">
        {isLoading ? (
          <div className="space-y-2 p-4">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
        ) : !employees.length ? (
          <EmptyState icon={<Users className="h-10 w-10" />} title="No employees yet"
            hint="Click “Add employee” to open the live camera and enrol faces directly from the RTSP stream." />
        ) : !filtered.length ? (
          <EmptyState icon={<Search className="h-10 w-10" />} title="No matches" hint={`Nothing matches “${search}”.`} />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs font-semibold uppercase tracking-wide text-muted">
                  <th className="px-4 py-3">Employee</th>
                  <th className="px-4 py-3">Code</th>
                  <th className="hidden px-4 py-3 md:table-cell">Department</th>
                  <th className="hidden px-4 py-3 lg:table-cell">Enrolled</th>
                  <th className="hidden px-4 py-3 sm:table-cell">Added</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((e) => (
                  <tr key={e.id} className="border-b border-[rgb(var(--border))] transition-colors last:border-0 hover:bg-[rgb(var(--surface-2))]">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <Avatar emp={e} />
                        <div className="min-w-0">
                          <p className="truncate font-semibold">{e.full_name}</p>
                          <p className="truncate text-xs text-muted">{e.job_title || "—"}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3"><span className="font-mono text-xs">{e.employee_code || "—"}</span></td>
                    <td className="hidden px-4 py-3 md:table-cell">{e.department || "—"}</td>
                    <td className="hidden px-4 py-3 lg:table-cell">
                      <Badge tone={e.embeddings > 0 ? "green" : "gray"}><ScanFace className="h-3 w-3" />{fmt(e.embeddings)} face{e.embeddings === 1 ? "" : "s"}</Badge>
                    </td>
                    <td className="hidden px-4 py-3 text-muted sm:table-cell">{timeAgo(e.created_at)}</td>
                    <td className="px-4 py-3">
                      <div className="flex justify-end gap-1.5">
                        <button className="btn-icon btn-ghost" title="Manage" onClick={() => openEdit(e)}><Pencil className="h-4 w-4" /></button>
                        <button className="btn-icon btn-ghost text-rose-500" title="Delete" onClick={() => setToDelete(e)}><Trash2 className="h-4 w-4" /></button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {filtered.length > PER_PAGE && (
          <div className="flex items-center justify-between border-t px-4 py-3 text-sm">
            <span className="text-muted">Showing {clampedPage * PER_PAGE + 1}–{Math.min((clampedPage + 1) * PER_PAGE, filtered.length)} of {filtered.length}</span>
            <div className="flex items-center gap-2">
              <button className="btn-icon btn-outline" disabled={clampedPage === 0} onClick={() => setPage(clampedPage - 1)}><ChevronLeft className="h-4 w-4" /></button>
              <span className="text-sm font-semibold">{clampedPage + 1} / {pages}</span>
              <button className="btn-icon btn-outline" disabled={clampedPage >= pages - 1} onClick={() => setPage(clampedPage + 1)}><ChevronRight className="h-4 w-4" /></button>
            </div>
          </div>
        )}
      </Card>

      {dialogOpen && (
        <EmployeeDialog open={dialogOpen} onClose={() => setDialogOpen(false)}
          onSaved={() => invalidate("employees", "dashboard")}
          cameras={camData?.cameras ?? []} activeCamera={camData?.active_camera ?? null} employee={editing} />
      )}
      <ConfirmDialog open={Boolean(toDelete)} onClose={() => setToDelete(null)} onConfirm={confirmDelete}
        loading={deleting} title="Delete employee"
        message={<>Delete <b>{toDelete?.full_name}</b>? This removes their images and face vectors permanently.</>} />
    </div>
  );
}

function Avatar({ emp }: { emp: Employee }) {
  const src = emp.profile_image ? mediaUrl(emp.profile_image) : emp.images[0] ? mediaUrl(emp.images[0].path) : null;
  const initials = emp.full_name.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();
  return (
    <div className={cx("grid h-10 w-10 shrink-0 place-items-center overflow-hidden rounded-full text-xs font-bold",
      src ? "" : "bg-brand-500/15 text-brand-400")}>
      {src ? <img src={src} className="h-full w-full object-cover" onError={(e) => { (e.target as HTMLImageElement).replaceWith(Object.assign(document.createElement("span"), { textContent: initials })); }} /> : initials || <ImageIcon className="h-4 w-4" />}
    </div>
  );
}
