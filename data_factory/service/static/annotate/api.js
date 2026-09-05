// Talking to the service. Thin, and honest about failure.
//
// The one piece of real logic is the 409 path: a save rejected because someone
// else moved the image comes back with the server's current state attached, so
// the caller can reconcile instead of telling a person their work is gone.

const KEY_HEADER = "X-API-Key";

export class ApiError extends Error {
  constructor(status, message, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

export class Conflict extends ApiError {
  constructor(message, current) {
    super(409, message, current);
    this.current = current;
  }
}

export class Api {
  constructor(base = "") {
    this.base = base;
    // Only ever set on a deployment that runs with a key; a local install
    // never sees this and never prompts for one.
    this.key = new URLSearchParams(location.search).get("key") || "";
  }

  headers(extra = {}) {
    const headers = { ...extra };
    if (this.key) headers[KEY_HEADER] = this.key;
    return headers;
  }

  async request(method, path, body) {
    const options = { method, headers: this.headers() };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await fetch(this.base + path, options);
    } catch (cause) {
      // A dropped connection is not a bug in the request; say so, because the
      // difference decides whether the user should retry or reload.
      throw new ApiError(0, "cannot reach the server", { cause: String(cause) });
    }

    if (response.status === 204) return null;

    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text };
      }
    }

    if (response.ok) return payload;

    const detail = payload && payload.detail;
    if (response.status === 409 && detail && detail.current) {
      throw new Conflict(detail.message || "this image changed", detail.current);
    }
    throw new ApiError(
      response.status,
      typeof detail === "string" ? detail : `request failed (${response.status})`,
      payload
    );
  }

  get(path) {
    return this.request("GET", path);
  }

  post(path, body) {
    return this.request("POST", path, body ?? {});
  }

  patch(path, body) {
    return this.request("PATCH", path, body);
  }

  put(path, body) {
    return this.request("PUT", path, body);
  }

  delete(path) {
    return this.request("DELETE", path);
  }

  // -- the calls the editor makes ------------------------------------------

  project(id) {
    return this.get(`/api/projects/${id}`);
  }

  projects() {
    return this.get("/api/projects");
  }

  createProject(name) {
    return this.post("/api/projects", { name });
  }

  images(id, { state = "", limit = 500, offset = 0 } = {}) {
    const query = new URLSearchParams({ state, limit, offset });
    return this.get(`/api/projects/${id}/images?${query}`);
  }

  image(projectId, imageId) {
    return this.get(`/api/projects/${projectId}/images/${imageId}`);
  }

  resume(projectId) {
    return this.get(`/api/projects/${projectId}/resume`);
  }

  saveShapes(projectId, imageId, version, shapes) {
    return this.put(`/api/projects/${projectId}/images/${imageId}/shapes`, {
      version,
      shapes,
    });
  }

  setState(projectId, imageId, state) {
    return this.patch(`/api/projects/${projectId}/images/${imageId}`, { state });
  }

  labels(projectId) {
    return this.get(`/api/projects/${projectId}/labels`);
  }

  createLabel(projectId, name) {
    return this.post(`/api/projects/${projectId}/labels`, { name });
  }

  renameLabel(projectId, idx, name) {
    return this.patch(`/api/projects/${projectId}/labels/${idx}`, { name });
  }

  retireLabel(projectId, idx) {
    return this.delete(`/api/projects/${projectId}/labels/${idx}`);
  }

  mergeLabel(projectId, idx, into) {
    return this.post(`/api/projects/${projectId}/labels/${idx}/merge`, { into });
  }

  summary(projectId) {
    return this.get(`/api/projects/${projectId}/summary`);
  }

  accept(projectId, imageId) {
    return this.post(`/api/projects/${projectId}/images/${imageId}/accept`);
  }

  runs(projectId) {
    return this.get(`/api/projects/${projectId}/runs`);
  }

  tools() {
    return this.get("/api/tools");
  }

  runTool(projectId, name, { imageId, params, label }) {
    return this.post(`/api/projects/${projectId}/tools/${name}`, {
      image_id: imageId,
      params,
      label,
    });
  }

  // Predictions arrive as a file, so this is multipart rather than JSON --
  // the client still never names a path.
  async importPredictions(projectId, file, { minConfidence = 0 } = {}) {
    const form = new FormData();
    form.append("file", file);
    form.append("min_confidence", String(minConfidence));
    const response = await fetch(
      `${this.base}/api/projects/${projectId}/predictions`,
      { method: "POST", headers: this.headers(), body: form }
    );
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload && payload.detail;
      throw new ApiError(
        response.status,
        typeof detail === "string" ? detail : "could not import those predictions",
        payload
      );
    }
    return payload;
  }

  imageUrl(projectId, imageId) {
    const key = this.key ? `?key=${encodeURIComponent(this.key)}` : "";
    return `${this.base}/api/projects/${projectId}/images/${imageId}/file${key}`;
  }

  thumbUrl(projectId, imageId) {
    const key = this.key ? `?key=${encodeURIComponent(this.key)}` : "";
    return `${this.base}/api/projects/${projectId}/images/${imageId}/thumb${key}`;
  }
}
