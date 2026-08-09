import styles from "./Sidebar.module.css";

interface SidebarProps {
  onReset: () => void;
  disabled: boolean;
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ onReset, disabled, open, onClose }: SidebarProps) {
  return (
    <>
      <aside className={`${styles.side} ${open ? styles.open : ""}`} role="complementary">
        <div className={styles.avatar} aria-hidden="true">菜</div>
        <h2 className={styles.name}>真白花音</h2>
        <div className={styles.role}>AI 复刻 · 纪念向</div>
        <span className={styles.status}><span className={styles.dot} />在线 · 深夜电台</span>
        <p className={styles.desc}>
          元气、温柔、笨拙但倔强。2020 年起在 B 站开播，2026-05-01 毕业。
          这里是粉丝为她搭建的 AI 纪念亭——她已不在，但声音还在。
        </p>
        <p className={styles.fans}>「每天都和白菜在一起」</p>
        <div className={styles.spacer} />
        <button type="button" className={styles.reset} onClick={onReset} disabled={disabled}>
          ↻ 清空对话
        </button>
        <p className={styles.note}>AI 复刻纪念项目 · 非本人</p>
      </aside>
      <div
        className={`${styles.overlay} ${open ? styles.overlayVisible : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />
    </>
  );
}
