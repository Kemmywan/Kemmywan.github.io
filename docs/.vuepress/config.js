module.exports = {
  title: "Kemmy的笔记本",
  description: "Kemmy的笔记本",

  // 网站 head 配置
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    // KaTeX 支持
    ['link', { rel: 'stylesheet', href: 'https://fastly.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css' }],
    // GitHub Markdown 样式（可选）
    ['link', { rel: 'stylesheet', href: 'https://fastly.jsdelivr.net/github-markdown-css/2.2.1/github-markdown.css' }],
    // viewport
    ['meta', { name: 'viewport', content: 'width=device-width,initial-scale=1,user-scalable=no' }]
  ],

  // 使用 reco 主题
  theme: 'reco',

  // 主题配置
  themeConfig: {
    // 导航栏
    nav: [
      { text: 'Home', link: '/', icon: 'reco-home' },
      { text: 'TimeLine', link: '/timeline/', icon: 'reco-date' },
      {
        text: 'Category',
        icon: 'reco-category',
        items: [
          { text: 'CUDA', link: '/categories/CUDA/' },
          { text: '密码学', link: '/categories/密码学/' },
          { text: '数学', link: '/categories/数学/' },
          { text: '工具', link: '/categories/工具/' }
        ]
      },
      { text: 'Tag', link: '/tag/', icon: 'reco-tag' },
      { text: 'About', link: 'https://github.com/<你的用户名>', icon: 'reco-account' }
    ],

    // 侧边栏（auto 模式，自动从标题生成）
    sidebar: 'auto',

    // 博客配置
    type: 'blog',
    blogConfig: {
      category: {
        location: 2,  // 在导航栏第几个位置
        text: 'Category'
      },
      tag: {
        location: 3,
        text: 'Tag'
      }
    },

    // 友链（可选）
    friendLink: [
      {
        title: 'vuepress-theme-reco',
        desc: 'A simple and beautiful vuepress Blog & Doc theme.',
        link: 'https://vuepress-theme-reco.recoluan.com'
      }
    ],

    // 个人信息
    author: 'Kemmy',
    authorAvatar: '/avatar.svg',

    // 搜索
    search: true,
    searchMaxSuggestions: 10,

    // 最后更新时间
    lastUpdated: 'Last Updated',

    // 仓库
    repo: '<你的用户名>/<你的用户名>.github.io',
    repoLabel: '查看源码',

    // 页脚
    startYear: '2026',

    // 评论（可选，先关闭）
    // valineConfig: { appId: '...', appKey: '...' }
  },

  // Markdown 插件
  markdown: {
    lineNumbers: true,
    extendMarkdown: md => {
      md.use(require('markdown-it-katex'))
    }
  },

  // 插件
  plugins: [
    ['@vuepress/medium-zoom']
  ]
}