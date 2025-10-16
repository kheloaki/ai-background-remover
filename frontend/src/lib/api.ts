export const API = process.env.NEXT_PUBLIC_API_BASE_URL || '';

interface ApiOptions {
  model?: string;
  padding?: number;
  white_bg?: boolean;
  enhance?: boolean;
  response_type?: 'base64' | 'url';
}

export async function removeBgFile(file: File, opts: ApiOptions) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('model', opts.model || 'u2net');
  fd.append('padding', String(opts.padding ?? 10));
  fd.append('white_bg', String(!!opts.white_bg));
  fd.append('enhance', String(!!opts.enhance));
  fd.append('response_type', opts.response_type || 'base64');
  const r = await fetch(API + '/api/remove-background', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function removeBgBase64(b64: string, opts: ApiOptions) {
  const r = await fetch(API + '/api/remove-background-base64', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: b64,
      model: opts.model || 'u2net',
      padding: opts.padding ?? 10,
      white_bg: !!opts.white_bg,
      enhance: !!opts.enhance,
      response_type: opts.response_type || 'base64',
    })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function removeBgUrl(url: string, opts: ApiOptions) {
  const r = await fetch(API + '/api/remove-background-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_url: url,
      model: opts.model || 'u2net',
      padding: opts.padding ?? 10,
      white_bg: !!opts.white_bg,
      enhance: !!opts.enhance,
      response_type: opts.response_type || 'base64',
    })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}