export interface QueueItem {
  key: string; // `${messageId}:${index}`
  url: string;
}

export function selectAutoPlayItem(
  items: readonly QueueItem[],
  activeKey: string | null,
  audioPaused: boolean,
): QueueItem | null {
  if (activeKey !== null || !audioPaused) return null;
  // Played clips stay available for manual replay, so index 0 can be stale.
  return items[items.length - 1] ?? null;
}
