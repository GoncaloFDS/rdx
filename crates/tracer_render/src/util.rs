pub trait Align<T> {
    fn align_up(self, value: T) -> Option<T>;
}

impl<T> Align<u64> for T
where
    T: Into<u64>,
{
    fn align_up(self, value: u64) -> Option<u64> {
        let align = self.into();
        Some(align.checked_add(value)? & !align)
    }
}

impl<T> Align<u32> for T
where
    T: Into<u32>,
{
    fn align_up(self, value: u32) -> Option<u32> {
        let align = self.into();
        Some(align.checked_add(value)? & !align)
    }
}

impl<T> Align<u16> for T
where
    T: Into<u16>,
{
    fn align_up(self, value: u16) -> Option<u16> {
        let align = self.into();
        Some(align.checked_add(value)? & !align)
    }
}

impl<T> Align<u8> for T
where
    T: Into<u8>,
{
    fn align_up(self, value: u8) -> Option<u8> {
        let align = self.into();
        Some(align.checked_add(value)? & !align)
    }
}

impl<T> Align<usize> for T
where
    T: Into<usize>,
{
    fn align_up(self, value: usize) -> Option<usize> {
        let align = self.into();
        Some(align.checked_add(value)? & !align)
    }
}

pub fn align_up<A, T>(align_mask: A, value: T) -> Option<T>
where
    A: Align<T>,
{
    align_mask.align_up(value)
}

pub fn align_down(align_mask: u64, value: u64) -> u64 {
    value & !align_mask
}
