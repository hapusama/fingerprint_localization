import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Periodic complex LFM chirp source for multipath sounding."""

    def __init__(
        self,
        sample_rate=500000.0,
        chirp_bw=200000.0,
        chirp_duration=0.008,
        guard_duration=0.002,
        repeat_pause=0.010,
        amplitude=0.05,
    ):
        gr.sync_block.__init__(
            self,
            name="Multipath Chirp Source",
            in_sig=None,
            out_sig=[np.complex64],
        )
        self.sample_rate = float(sample_rate)
        self.chirp_bw = float(chirp_bw)
        self.chirp_duration = float(chirp_duration)
        self.guard_duration = float(guard_duration)
        self.repeat_pause = float(repeat_pause)
        self.amplitude = float(amplitude)
        self._index = 0
        self._make_waveform()

    def _make_waveform(self):
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.chirp_duration <= 0:
            raise ValueError("chirp_duration must be positive")
        if self.chirp_bw <= 0 or self.chirp_bw >= self.sample_rate:
            raise ValueError("chirp_bw must be positive and smaller than sample_rate")

        n_chirp = max(32, int(round(self.sample_rate * self.chirp_duration)))
        n_guard = max(0, int(round(self.sample_rate * self.guard_duration)))
        n_pause = max(0, int(round(self.sample_rate * self.repeat_pause)))

        t = np.arange(n_chirp, dtype=np.float64) / self.sample_rate
        sweep_rate = self.chirp_bw / self.chirp_duration
        phase = 2.0 * np.pi * ((-self.chirp_bw / 2.0) * t + 0.5 * sweep_rate * t * t)
        chirp = np.exp(1j * phase).astype(np.complex64)
        chirp *= np.hanning(n_chirp).astype(np.float32)
        chirp *= np.float32(self.amplitude)

        guard = np.zeros(n_guard, dtype=np.complex64)
        pause = np.zeros(n_pause, dtype=np.complex64)
        waveform = np.concatenate((guard, chirp, guard, pause))
        if waveform.size == 0:
            waveform = np.zeros(1, dtype=np.complex64)
        self._waveform = waveform.astype(np.complex64, copy=False)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        idx = (np.arange(n, dtype=np.int64) + self._index) % self._waveform.size
        out[:] = self._waveform[idx]
        self._index = (self._index + n) % self._waveform.size
        return n
