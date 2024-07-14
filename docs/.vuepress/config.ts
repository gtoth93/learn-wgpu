import {defaultTheme, DefaultThemeOptions} from '@vuepress/theme-default'
import {defineUserConfig, UserConfig} from 'vuepress'
import {viteBundler, ViteBundlerOptions} from '@vuepress/bundler-vite'
import {searchPlugin} from "@vuepress/plugin-search";
import {registerComponentsPlugin, RegisterComponentsPluginOptions} from "@vuepress/plugin-register-components";
import wasm from "vite-plugin-wasm";
import * as path from "node:path";
import topLevelAwait from "vite-plugin-top-level-await";

const themeOptions: DefaultThemeOptions = {
    hostname: 'https://sotrh.github.io',
    contributors: false,
    themePlugins: {
        seo: {
            hostname: 'https://sotrh.github.io',
            author: {
                name: 'Benjamin Hansen',
                url: 'https://twitter.com/sotrh760',
            }
        }
    },
    sidebar: [
        {
            text: 'Introduction',
            link: '/'
        },
        {
            text: 'Beginner',
            collapsible: false,
            children: [
                '/beginner/tutorial1-window/',
                '/beginner/tutorial2-surface/',
                '/beginner/tutorial3-pipeline/',
                '/beginner/tutorial4-buffer/',
                '/beginner/tutorial5-textures/',
                '/beginner/tutorial6-uniforms/',
                '/beginner/tutorial7-instancing/',
                '/beginner/tutorial8-depth/',
                '/beginner/tutorial9-models/',
            ],
        },
        {
            text: 'Intermediate',
            collapsible: false,
            children: [
                '/intermediate/tutorial10-lighting/',
                '/intermediate/tutorial11-normals/',
                '/intermediate/tutorial12-camera/',
                '/intermediate/tutorial13-hdr/',
            ],
        },
        {
            text: 'Showcase',
            collapsible: true,
            children: [
                '/showcase/',
                '/showcase/windowless/',
                '/showcase/gifs/',
                '/showcase/pong/',
                '/showcase/compute/',
                '/showcase/alignment/',
            ]
        },
        {
            text: 'News',
            collapsible: true,
            children: [
                '/news/0.18 and hdr/',
                '/news/0.17/',
                '/news/0.16/',
                '/news/0.15/',
                '/news/0.14/',
                '/news/0.13/',
                '/news/0.12/',
                '/news/pre-0.12/',
            ]
        }
    ]
};

const registerComponentsOptions: RegisterComponentsPluginOptions = {
    componentsDir: path.resolve(__dirname, './components')
};

const bundlerOptions: ViteBundlerOptions = {
    viteOptions: {
        plugins: [
            wasm(),
            topLevelAwait()
        ]
    }
};

const userConfig: UserConfig = {
    lang: 'en-US',
    title: 'Learn Wgpu',
    base: '/learn-wgpu/',

    theme: defaultTheme(themeOptions),

    plugins: [
        registerComponentsPlugin(registerComponentsOptions),
        searchPlugin()
    ],

    bundler: viteBundler(bundlerOptions),
};

export default defineUserConfig(userConfig)
