// ResizableDrawer uses Modal z-index 1400; MUI Dialog default is 1300, so dialogs would stack under the drawer.
export const DIALOG_Z_INDEX_ABOVE_STEP_DRAWER = 1600;
// Select Menu uses modal z-index (1300) by default; it must sit above the step drawer dialogs.
const SELECT_MENU_Z_INDEX_ABOVE_STEP_DIALOG = DIALOG_Z_INDEX_ABOVE_STEP_DRAWER + 100;
// Select menus must sit above the drawer Modal (1400) or they appear behind the pane.
const SELECT_MENU_Z_ABOVE_STEP_DRAWER = 1450;

/** Use on MUI Select `MenuProps` inside StepConfigDrawer / ResizableDrawer so menus are visible and clickable. */
export const STEP_DRAWER_SELECT_MENU_PROPS = {
  disableScrollLock: true,
  sx: { zIndex: SELECT_MENU_Z_ABOVE_STEP_DRAWER },
  PaperProps: {
    sx: { zIndex: SELECT_MENU_Z_ABOVE_STEP_DRAWER },
  },
};

/** Selects inside dialogs opened from the step drawer need an even higher z-index. */
export const STEP_DIALOG_SELECT_MENU_PROPS = {
  disableScrollLock: true,
  sx: { zIndex: SELECT_MENU_Z_INDEX_ABOVE_STEP_DIALOG },
  PaperProps: {
    sx: { zIndex: SELECT_MENU_Z_INDEX_ABOVE_STEP_DIALOG },
  },
};
