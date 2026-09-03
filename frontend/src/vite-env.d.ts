/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_USE_MOCKS?: string;
  readonly VITE_MOCK_SPEED?: string;
  readonly VITE_MAX_UPLOAD_BYTES?: string;
  readonly VITE_ENABLE_BASEMAP?: string;
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
