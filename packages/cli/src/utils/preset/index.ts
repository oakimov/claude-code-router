/**
 * Preset functionality CLI layer
 * Exports all preset-related functions and types
 */

// Re-export types and core functions from shared package
export * from '@CCR/shared';

// Export CLI specific functions (with interaction)
export { exportPresetCli } from './export';
export { installPresetCli, applyPresetCli } from './install';
export { handlePresetCommand } from './commands';
