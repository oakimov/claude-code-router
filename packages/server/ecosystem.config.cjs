module.exports = {
  apps: [
    {
      name: 'claude-code-router-server',
      script: '/app/packages/server/dist/index.js',
      cwd: '/app/packages/server',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        // @cursor/sdk uses node:sqlite for its local agent store; silence the
        // ExperimentalWarning it prints on first use (verified flag on Node 22+).
        NODE_OPTIONS: '--disable-warning=ExperimentalWarning',
      },
      // 日志配置
      error_file: '/root/.claude-code-router/logs/error.log',
      out_file: '/root/.claude-code-router/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      // Enable log timestamps
      time: true,
    },
  ],
};
