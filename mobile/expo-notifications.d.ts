/**
 * Minimal typings for expo-notifications (CI / environments without node_modules).
 * When dependencies are installed, package types take precedence if present.
 */
declare module 'expo-notifications' {
  export interface NotificationContent {
    data?: Record<string, unknown>;
  }

  export interface NotificationRequest {
    content: NotificationContent;
  }

  export interface Notification {
    request: NotificationRequest;
  }

  export interface NotificationResponse {
    notification: Notification;
  }

  export type NotificationHandler = (notification: unknown) => Promise<{
    shouldShowAlert: boolean;
    shouldPlaySound: boolean;
    shouldSetBadge: boolean;
  }>;

  export function setNotificationHandler(handler: { handleNotification: NotificationHandler }): void;

  export function getPermissionsAsync(): Promise<{ status: string }>;
  export function requestPermissionsAsync(): Promise<{ status: string }>;

  export function getExpoPushTokenAsync(options?: { projectId?: string }): Promise<{ data: string }>;

  export function addNotificationResponseReceivedListener(
    listener: (response: NotificationResponse) => void
  ): { remove(): void };
}
