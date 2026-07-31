import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

function HomepageHeader() {
  const {siteConfig, i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      title: 'Route Claude Code',
      highlight: 'to Any LLM',
      subtitle: 'Use Claude Code without an Anthropic account. Connect to DeepSeek, Gemini, Groq, and more.',
      getStarted: 'Get Started',
      github: 'View on GitHub',
    },
    'zh-CN': {
      title: '将 Claude Code',
      highlight: '路由到任何 LLM',
      subtitle: '无需 Anthropic 账户即可使用 Claude Code。支持连接 DeepSeek、Gemini、Groq 等提供商。',
      getStarted: '开始使用',
      github: '查看 GitHub',
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <header className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-gradient-to-b from-gray-50 to-white">
      {/* Background Elements */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(rgba(204, 124, 94, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(204, 124, 94, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      ></div>

      {/* Hero Content */}
      <div className="relative z-10 container mx-auto px-4 py-20">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="animate-fade-in-up inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-full text-sm font-semibold mb-8">
            <span className="animate-pulse">●</span>
            <span>{currentLocale === 'zh-CN' ? '开源免费' : 'Open Source'}</span>
          </div>

          {/* Title */}
          <Heading
            as="h1"
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold mb-6 leading-tight animate-fade-in-up"
            style={{ animationDelay: '0.1s' }}
          >
            <span className="text-gray-900">Route </span>
            <span className="text-primary">Claude Code</span>
            <br />
            <span className="text-gray-900">{t.highlight}</span>
          </Heading>

          {/* Subtitle */}
          <p
            className="text-lg sm:text-xl md:text-2xl text-gray-600 mb-10 leading-relaxed max-w-3xl mx-auto animate-fade-in-up"
            style={{ animationDelay: '0.2s' }}
          >
            {t.subtitle}
          </p>

          {/* CTA Buttons */}
          <div
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12 animate-fade-in-up"
            style={{ animationDelay: '0.3s' }}
          >
            <Link
              className="px-6 sm:px-8 py-3 sm:py-4 bg-primary text-white hover:text-white rounded-lg text-base sm:text-lg font-semibold transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105"
              to="/docs/cli/quick-start"
            >
              {t.getStarted}
            </Link>
            <Link
              className="px-6 sm:px-8 py-3 sm:py-4 bg-white hover:bg-gray-50 text-gray-900 border border-gray-300 rounded-lg text-base sm:text-lg font-semibold transition-all duration-300 hover:border-primary hover:scale-105"
              to="https://github.com/oakimov/claude-code-router"
            >
              {t.github}
            </Link>
          </div>

          {/* Install Command */}
          <div
            className="animate-fade-in-up max-w-3xl mx-auto mt-16"
            style={{ animationDelay: '0.4s' }}
          >
            <div
              className="rounded-xl overflow-hidden"
              style={{
                background: 'linear-gradient(145deg, #1e1e1e 0%, #0d0d0d 100%)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1)',
              }}
            >
              {/* Terminal Header */}
              <div
                className="px-5 py-3 flex items-center justify-between"
                style={{
                  background: 'linear-gradient(180deg, #2a2a2a 0%, #1f1f1f 100%)',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                }}
              >
                <div className="flex items-center gap-2">
                  <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 transition-colors"></div>
                  </div>
                  <span className="ml-4 text-xs text-gray-400 font-mono flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M2 5a2 2 0 012-2h12a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V5zm3.293 1.293a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 01-1.414-1.414L7.586 10 5.293 7.707a1 1 0 010-1.414zM11 12a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                    </svg>
                    terminal
                  </span>
                </div>

                {/* Copy Button */}
                <button
                  onClick={() => navigator.clipboard.writeText('git clone https://github.com/oakimov/claude-code-router.git && cd claude-code-router/packages/server && docker compose up --build -d')}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono transition-all duration-200 hover:bg-white/10"
                  style={{ color: '#9ca3af' }}
                  title="Copy to clipboard"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <span>Copy</span>
                </button>
              </div>

              {/* Terminal Body */}
              <div className="p-8" style={{ background: '#0d0d0d' }}>
                <div className="font-mono text-base md:text-lg leading-relaxed text-left">
                  {/* Prompt */}
                  <div className="flex items-start gap-3">
                    <span style={{ color: '#22c55e', fontWeight: '600' }}>$</span>
                    <span className="flex-1">
                      <span style={{ color: '#60a5fa' }}>cd</span>
                      <span style={{ color: '#e5e7eb', marginLeft: '0.5rem' }}>
                        packages/server
                      </span>
                    </span>
                  </div>
                  <div className="flex items-start gap-3 mt-2">
                    <span style={{ color: '#22c55e', fontWeight: '600' }}>$</span>
                    <span className="flex-1">
                      <span style={{ color: '#60a5fa' }}>docker</span>
                      <span style={{ color: '#f97583', marginLeft: '0.5rem' }}>compose</span>
                      <span style={{ color: '#fbbf24', marginLeft: '0.5rem' }}>up</span>
                      <span style={{ color: '#f97583', marginLeft: '0.5rem' }}>--build</span>
                      <span style={{ color: '#e5e7eb', marginLeft: '0.5rem' }}>-d</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </header>
  );
}

function FeatureSection() {
  const {i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      title: 'Why Claude Code Router?',
      features: [
        {
          emoji: '⚡',
          title: 'Lightning Fast',
          description: 'Get started in minutes with just one command. No complicated configuration required.'
        },
        {
          emoji: '🎯',
          title: 'Smart Routing',
          description: 'Automatically route requests to the best model based on context length, task type, and custom rules.'
        },
        {
          emoji: '🔌',
          title: 'Multi-Provider',
          description: 'Support for DeepSeek, Gemini, Groq, OpenRouter, and more. Easy to extend with custom transformers.'
        },
        {
          emoji: '💰',
          title: 'Cost Effective',
          description: 'Use more affordable models for routine tasks while reserving Claude for complex scenarios.'
        },
        {
          emoji: '🛠️',
          title: 'Agent System',
          description: 'Extendable agent architecture for custom tools and workflows. Built-in support for image tasks.'
        },
        {
          emoji: '🔧',
          title: 'Highly Customizable',
          description: 'Configure routing per project, set up transformers, and fine-tune every aspect of your workflow.'
        }
      ]
    },
    'zh-CN': {
      title: '为什么选择 Claude Code Router？',
      features: [
        {
          emoji: '⚡',
          title: '快速上手',
          description: '只需一条命令即可开始使用，无需复杂配置。'
        },
        {
          emoji: '🎯',
          title: '智能路由',
          description: '基于上下文长度、任务类型和自定义规则，自动将请求路由到最佳模型。'
        },
        {
          emoji: '🔌',
          title: '多提供商支持',
          description: '支持 DeepSeek、Gemini、Groq、OpenRouter 等多个提供商，易于扩展。'
        },
        {
          emoji: '💰',
          title: '节省成本',
          description: '常规任务使用更经济的模型，复杂场景再使用 Claude。'
        },
        {
          emoji: '🛠️',
          title: 'Agent 系统',
          description: '可扩展的 agent 架构，支持自定义工具和工作流。内置图像任务支持。'
        },
        {
          emoji: '🔧',
          title: '高度可定制',
          description: '按项目配置路由、设置转换器，微调工作流的每个细节。'
        }
      ]
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <section className="py-16 sm:py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12 sm:mb-16">
          <Heading as="h2" className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            {t.title}
          </Heading>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 max-w-7xl mx-auto">
          {t.features.map((feature, idx) => (
            <div
              key={idx}
              className="group p-6 sm:p-8 bg-white rounded-2xl border border-gray-200 hover:border-primary transition-all duration-300 hover:shadow-2xl hover:-translate-y-2"
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              <div className="text-4xl sm:text-5xl mb-4">{feature.emoji}</div>
              <Heading as="h3" className="text-lg sm:text-xl font-bold text-gray-900 mb-3">
                {feature.title}
              </Heading>
              <p className="text-gray-600 leading-relaxed text-sm sm:text-base">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CodeDemo() {
  const {i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      title: 'Simple Configuration',
      subtitle: 'Configure your providers and routing logic with a single JSON file.'
    },
    'zh-CN': {
      title: '简单配置',
      subtitle: '使用单个 JSON 文件配置提供商和路由逻辑。'
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <section className="py-16 sm:py-24 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12 sm:mb-16">
          <Heading as="h2" className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            {t.title}
          </Heading>
          <p className="text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto">
            {t.subtitle}
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div
            className="rounded-2xl overflow-hidden shadow-2xl"
            style={{
              background: '#1a1a1a',
              border: '2px solid #374151'
            }}
          >
            <div
              className="px-6 py-4 flex items-center gap-2"
              style={{ background: '#2d2d2d', borderBottom: '1px solid #374151' }}
            >
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="ml-4 text-sm text-gray-300 font-mono">config.json</span>
            </div>
            <pre className="p-8 overflow-x-auto text-sm md:text-base" style={{ background: '#1a1a1a' }}>
              <code className="font-mono" style={{ lineHeight: '1.6' }}>
                <span style={{ color: '#f97583' }}>{'{'}</span>
                <span style={{ color: '#e5e7eb' }}>
                  {'\n  '}
                  <span style={{ color: '#79c0ff' }}>"Providers"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#f97583' }}>[</span>
                  {'\n    '}
                  <span style={{ color: '#f97583' }}>{'{'}</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"NAME"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"deepseek"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"HOST"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"https://api.deepseek.com"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"APIKEY"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"your-api-key"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"MODELS"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#f97583' }}>[</span>
                  <span style={{ color: '#a5d6ff' }}>"deepseek-chat"</span>
                  <span style={{ color: '#f97583' }}>, </span>
                  <span style={{ color: '#a5d6ff' }}>"deepseek-coder"</span>
                  <span style={{ color: '#f97583' }}>]</span>
                  {'\n    '}
                  <span style={{ color: '#f97583' }}>{'}'}</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n    '}
                  <span style={{ color: '#f97583' }}>{'{'}</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"NAME"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"groq"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"HOST"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"https://api.groq.com/openai/v1"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"APIKEY"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"your-groq-key"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n      '}
                  <span style={{ color: '#79c0ff' }}>"MODELS"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#f97583' }}>[</span>
                  <span style={{ color: '#a5d6ff' }}>"llama-3.3-70b-versatile"</span>
                  <span style={{ color: '#f97583' }}>]</span>
                  {'\n    '}
                  <span style={{ color: '#f97583' }}>{'}'}</span>
                  {'\n  '}
                  <span style={{ color: '#f97583' }}>]</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n  '}
                  <span style={{ color: '#79c0ff' }}>"Router"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#f97583' }}>{'{'}</span>
                  {'\n    '}
                  <span style={{ color: '#79c0ff' }}>"default"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"deepseek,deepseek-chat"</span>
                  <span style={{ color: '#f97583' }}>,</span>
                  {'\n    '}
                  <span style={{ color: '#79c0ff' }}>"background"</span>
                  <span style={{ color: '#f97583' }}>: </span>
                  <span style={{ color: '#a5d6ff' }}>"groq,llama-3.3-70b-versatile"</span>
                  {'\n  '}
                  <span style={{ color: '#f97583' }}>{'}'}</span>
                  {'\n'}
                  <span style={{ color: '#f97583' }}>{'}'}</span>
                </span>
              </code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

function UseCases() {
  const {i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      title: 'Perfect For',
      subtitle: 'See how Claude Code Router fits your workflow',
      cases: [
        {
          icon: '💰',
          title: 'Cost-Conscious Developers',
          description: 'Reduce API costs by 10x while maintaining quality for most tasks'
        },
        {
          icon: '🔒',
          title: 'Privacy-Focused Teams',
          description: 'Keep code local with self-hosted models while using Claude when needed'
        },
        {
          icon: '🔄',
          title: 'Multi-Model Workflows',
          description: 'Use different models for different tasks without switching tools'
        },
        {
          icon: '⚡',
          title: 'Claude Code Power Users',
          description: 'Extend Claude Code with custom providers and routing strategies'
        }
      ]
    },
    'zh-CN': {
      title: '适用场景',
      subtitle: '看看 Claude Code Router 如何融入您的工作流程',
      cases: [
        {
          icon: '💰',
          title: '注重成本的开发者',
          description: '在大多数任务上保持质量的同时，将 API 成本降低 10 倍'
        },
        {
          icon: '🔒',
          title: '注重隐私的团队',
          description: '使用自托管模型保持代码本地化，需要时再使用 Claude'
        },
        {
          icon: '🔄',
          title: '多模型工作流',
          description: '为不同任务使用不同模型，无需切换工具'
        },
        {
          icon: '⚡',
          title: 'Claude Code 高级用户',
          description: '使用自定义提供商和路由策略扩展 Claude Code'
        }
      ]
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <section className="py-16 sm:py-24 bg-gradient-to-b from-white to-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12 sm:mb-16">
          <Heading as="h2" className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            {t.title}
          </Heading>
          <p className="text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto">
            {t.subtitle}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 max-w-5xl mx-auto">
          {t.cases.map((useCase, idx) => (
            <div
              key={idx}
              className="group relative p-6 sm:p-8 bg-white rounded-2xl border-2 border-gray-100 hover:border-primary transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 overflow-hidden"
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              {/* Background decoration */}
              <div
                className="absolute top-0 right-0 w-32 h-32 rounded-full opacity-5 group-hover:opacity-10 transition-opacity"
                style={{
                  background: 'linear-gradient(135deg, #CC7C5E 0%, transparent 70%)',
                  transform: 'translate(30%, -30%)'
                }}
              ></div>

              {/* Number badge */}
              <div className="absolute top-4 sm:top-6 right-4 sm:right-6 w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center rounded-full text-base sm:text-lg font-bold transition-all duration-300 group-hover:scale-110"
                style={{
                  background: 'linear-gradient(135deg, #CC7C5E 0%, #BC5C3E 100%)',
                  color: 'white',
                  boxShadow: '0 4px 12px rgba(204, 124, 94, 0.3)'
                }}
              >
                {idx + 1}
              </div>

              {/* Icon */}
              <div className="text-4xl sm:text-5xl mb-4 group-hover:scale-110 transition-transform duration-300">
                {useCase.icon}
              </div>

              {/* Content */}
              <div>
                <Heading as="h3" className="text-lg sm:text-xl font-bold text-gray-900 mb-3 group-hover:text-primary transition-colors">
                  {useCase.title}
                </Heading>
                <p className="text-gray-600 leading-relaxed text-sm sm:text-base">
                  {useCase.description}
                </p>
              </div>

              {/* Hover accent line */}
              <div
                className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-primary to-primary/70 transition-all duration-300 group-hover:w-full"
                style={{ width: '0%' }}
              ></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  const {i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      title: 'Ready to Get Started?',
      subtitle: 'Join thousands of developers using Claude Code Router to build better software.',
      getStarted: 'Get Started',
      docs: 'Read the Docs',
      github: 'Star on GitHub',
      community: 'Join Community'
    },
    'zh-CN': {
      title: '准备开始了吗？',
      subtitle: '加入数千名使用 Claude Code Router 的开发者，构建更好的软件。',
      getStarted: '开始使用',
      docs: '阅读文档',
      github: '在 GitHub 上点赞',
      community: '加入社区'
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <section className="relative py-20 sm:py-32 overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-primary via-primary/90 to-primary/80"></div>
        {/* Pattern overlay */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `
              radial-gradient(circle at 25% 25%, white 1%, transparent 1%),
              radial-gradient(circle at 75% 75%, white 1%, transparent 1%)
            `,
            backgroundSize: '40px 40px'
          }}
        ></div>
        {/* Floating orbs */}
        <div className="absolute top-20 left-10 w-64 h-64 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      <div className="relative container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          {/* Main title */}
          <Heading as="h2" className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
            {t.title}
          </Heading>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl md:text-2xl text-white/95 mb-10 sm:mb-12 max-w-3xl mx-auto leading-relaxed">
            {t.subtitle}
          </p>

          {/* Primary CTA buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8 sm:mb-12">
            <Link
              className="group px-8 sm:px-10 py-4 sm:py-5 bg-white text-primary hover:bg-gray-50 rounded-xl text-base sm:text-lg font-bold transition-all duration-300 shadow-2xl hover:shadow-white/20 hover:scale-105 flex items-center justify-center gap-2"
              to="/docs/cli/quick-start"
            >
              {t.getStarted}
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              className="px-8 sm:px-10 py-4 sm:py-5 bg-white/10 backdrop-blur-sm text-white border-2 border-white/30 hover:bg-primary hover:border-primary hover:text-white rounded-xl text-base sm:text-lg font-bold transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              to="/docs/category/cli"
            >
              {t.docs}
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </Link>
          </div>

          {/* Secondary links */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center text-white/80">
            <a
              href="https://github.com/oakimov/claude-code-router"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 hover:text-white transition-colors group"
            >
              <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
              <span className="text-sm font-medium">{t.github}</span>
            </a>

            <div className="hidden sm:block w-px h-4 bg-white/30"></div>

            <a
              href="https://github.com/oakimov/claude-code-router/discussions"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 hover:text-white transition-colors group"
            >
              <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              <span className="text-sm font-medium">{t.community}</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

function ForkCredit() {
  const {i18n} = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  const content = {
    en: {
      text: 'This is an active fork of',
      original: 'musistudio/claude-code-router',
      description: 'Contains additional provider integrations, bug fixes, and improvements not yet merged upstream.',
    },
    'zh-CN': {
      text: '这是一个更活跃的分支，基于',
      original: 'musistudio/claude-code-router',
      description: '包含上游尚未合并的额外提供商集成、错误修复和改进。',
    }
  };

  const t = content[currentLocale as keyof typeof content] || content.en;

  return (
    <section className="py-12 bg-gray-50 border-t border-gray-200">
      <div className="container mx-auto px-4 text-center">
        <p className="text-gray-500 text-sm">
          {t.text}{' '}
          <a
            href="https://github.com/musistudio/claude-code-router"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline font-medium"
          >
            {t.original}
          </a>
        </p>
        <p className="text-gray-400 text-xs mt-2">{t.description}</p>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout>
      <HomepageHeader />
      <main>
        <FeatureSection />
        <CodeDemo />
        <UseCases />
        <CTASection />
      </main>
      <ForkCredit />
    </Layout>
  );
}
