import { useRef, useCallback } from 'react';

/**
 * Hook for detecting long-press gestures on touch devices
 * Works alongside onContextMenu for desktop support
 * 
 * @param {Function} onLongPress - Callback when long-press is detected
 * @param {Object} options - Configuration options
 * @param {number} options.delay - Delay in ms before long-press is detected (default: 600)
 * @param {number} options.moveThreshold - Maximum pixel movement allowed before canceling (default: 10)
 * @returns {Object} Touch event handlers to attach to elements
 */
export const useLongPress = (onLongPress, options = {}) => {
  const { delay = 600, moveThreshold = 10 } = options;
  const timeoutRef = useRef(null);
  const startPosRef = useRef(null);
  const isLongPressRef = useRef(false);

  const start = useCallback((event) => {
    if (event.touches && event.touches.length > 0) {
      const touch = event.touches[0];
      startPosRef.current = {
        x: touch.clientX,
        y: touch.clientY
      };
      isLongPressRef.current = false;

      timeoutRef.current = setTimeout(() => {
        isLongPressRef.current = true;
        if (onLongPress) {
          // Create a synthetic event similar to contextmenu event
          const syntheticEvent = {
            preventDefault: () => event.preventDefault(),
            clientX: touch.clientX,
            clientY: touch.clientY,
            touches: event.touches,
            target: event.target
          };
          onLongPress(syntheticEvent);
        }
      }, delay);
    }
  }, [onLongPress, delay]);

  const move = useCallback((event) => {
    if (!startPosRef.current || !event.touches || event.touches.length === 0) return;

    const touch = event.touches[0];
    const deltaX = Math.abs(touch.clientX - startPosRef.current.x);
    const deltaY = Math.abs(touch.clientY - startPosRef.current.y);

    // Cancel if moved too far
    if (deltaX > moveThreshold || deltaY > moveThreshold) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      startPosRef.current = null;
    }
  }, [moveThreshold]);

  const end = useCallback((event) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    
    // If it was a long press, prevent the default click behavior
    if (isLongPressRef.current) {
      event.preventDefault();
      event.stopPropagation();
    }
    
    startPosRef.current = null;
    isLongPressRef.current = false;
  }, []);

  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    startPosRef.current = null;
    isLongPressRef.current = false;
  }, []);

  return {
    onTouchStart: start,
    onTouchMove: move,
    onTouchEnd: end,
    onTouchCancel: cancel
  };
};

