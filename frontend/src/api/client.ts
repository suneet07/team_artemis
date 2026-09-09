import type { ApiError, ApiErrorEnvelope } from "@contracts/types";

export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

/**
 * Mocks are opt-in for a built bundle, and on by default only while developing.
 *
 * This was `!== "0"`, which meant a production build with no environment
 * configured served MSW fixtures: fabricated answers, fabricated imagery, and
 * a console that looked entirely healthy while showing none of the real
 * system. A deployment that forgets one variable should fail visibly, not
 * quietly invent data.
 *
 * `import.meta.env.DEV` is compile-time, so `npm run dev` keeps mocks without
 * anyone setting anything, and `vite build` cannot switch them on by accident.
 */
export const USE_MOCKS: boolean =
  import.meta.env.VITE_USE_MOCKS === "1" ||
  (import.meta.env.DEV && import.meta.env.VITE_USE_MOCKS !== "0");

export const MAX_UPLOAD_BYTES: number = Number(
  import.meta.env.VITE_MAX_UPLOAD_BYTES ?? 4 * 1024 * 1024 * 1024,
);

export const ENABLE_BASEMAP: boolean =
  import.meta.env.VITE_ENABLE_BASEMAP === "1";

/** Reserved slot. Phase 1 has no auth; adding it later is this one line. */
const API_KEY: string | undefined = import.meta.env.VITE_API_KEY;

export class ApiFailure extends Error {
  readonly status: number;
  readonly error: ApiError;

  constructor(status: number, error: ApiError) {
    super(error.message);
    this.name = "ApiFailure";
    this.status = status;
    this.error = error;
  }
}

function authHeaders(): Record<string, string> {
  return API_KEY ? { "X-Api-Key": API_KEY } : {};
}

/** Absolute URL for a path the API returned (`/api/v1/...`) or a bare route. */
export function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//.test(pathOrUrl)) return pathOrUrl;
  if (pathOrUrl.startsWith("/api/")) {
    const base = new URL(API_BASE);
    return `${base.origin}${pathOrUrl}`;
  }
  return `${API_BASE}${pathOrUrl.startsWith("/") ? "" : "/"}${pathOrUrl}`;
}

async function toFailure(response: Response): Promise<ApiFailure> {
  let error: ApiError = {
    code: "INTERNAL",
    message: `Request failed with status ${response.status}.`,
  };
  try {
    const body = (await response.json()) as Partial<ApiErrorEnvelope>;
    if (body && typeof body === "object" && body.error) error = body.error;
  } catch {
    /* non-JSON error body — keep the generic envelope */
  }
  return new ApiFailure(response.status, error);
}

export async function apiGet<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(resolveUrl(path), {
    ...init,
    method: "GET",
    headers: { Accept: "application/json", ...authHeaders(), ...init?.headers },
  });
  if (!response.ok) throw await toFailure(response);
  return (await response.json()) as T;
}

export async function apiSend<T>(
  method: "POST" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(resolveUrl(path), {
    method,
    headers: {
      Accept: "application/json",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...authHeaders(),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await toFailure(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface UploadHandle {
  promise: Promise<unknown>;
  abort: () => void;
}

/**
 * Scene upload.
 *
 * XHR rather than fetch because we need a byte-level progress readout on
 * multi-gigabyte rasters. The transport is deliberately isolated here: if the
 * backend's cap forces a chunked `POST /scenes/init` + `PUT .../parts/{n}`
 * pair later, only this function changes.
 */
export function uploadScene(
  file: File,
  fields: Record<string, string>,
  onProgress: (loadedBytes: number, totalBytes: number) => void,
): UploadHandle {
  const form = new FormData();
  form.append("file", file);
  for (const [key, value] of Object.entries(fields)) form.append(key, value);

  const xhr = new XMLHttpRequest();
  const promise = new Promise<unknown>((resolve, reject) => {
    xhr.open("POST", resolveUrl("/scenes"));
    for (const [key, value] of Object.entries(authHeaders())) {
      xhr.setRequestHeader(key, value);
    }
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded, event.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(
            new ApiFailure(xhr.status, {
              code: "INTERNAL",
              message: "The upload response could not be read.",
            }),
          );
        }
        return;
      }
      let error: ApiError = {
        code: "INTERNAL",
        message: `Upload failed with status ${xhr.status}.`,
      };
      try {
        const parsed = JSON.parse(xhr.responseText) as ApiErrorEnvelope;
        if (parsed?.error) error = parsed.error;
      } catch {
        /* keep the generic envelope */
      }
      reject(new ApiFailure(xhr.status, error));
    });
    xhr.addEventListener("error", () =>
      reject(
        new ApiFailure(0, {
          code: "INTERNAL",
          message: "The upload could not reach the API.",
          hint: "Check that the backend is running and reachable.",
        }),
      ),
    );
    xhr.addEventListener("abort", () =>
      reject(
        new ApiFailure(0, {
          code: "JOB_CANCELLED",
          message: "Upload cancelled.",
        }),
      ),
    );
    xhr.send(form);
  });

  return { promise, abort: () => xhr.abort() };
}
