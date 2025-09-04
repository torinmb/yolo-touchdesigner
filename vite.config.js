import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  plugins: [
    viteStaticCopy({
      targets: [
        // Copy WASM binaries
        { src: 'node_modules/onnxruntime-web/dist/*.wasm', dest: './' },
        // Copy the dynamic .mjs helpers (this is the missing piece)
        { src: 'node_modules/onnxruntime-web/dist/*.mjs', dest: './' },
      ],
    }),
  ],
  assetsInclude: ['**/*.onnx'],
  base: './',
})
