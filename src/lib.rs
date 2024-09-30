use core::fmt;

use ordered_float::FloatCore;
use ordered_float::NotNan;

use rust_string_random::random;
use rust_string_random::Options;
use rust_string_random::RandWay;

use std::cmp::Ordering;

use std::fmt::Debug;
use std::fmt::Display;

use std::hash::Hash;
use std::hash::Hasher;

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::ops::SubAssign;

//TODO: possible optimization is to remove NotNaN and use vectors to store children
//if it's evaluated in such a manner where the possible children are either 1 or 2
//there's no need for NotNan to make floats hashable
//TODO: add graphviz support

const RANDOM_STRING_CONFIG: Options = Options {
    rand: RandWay::NORMAL,
    numbers: None,
    letters: None,
    specials: None,
};

trait SmallgradFloat: FloatCore + Debug {}
impl<T: FloatCore + Debug> SmallgradFloat for T {}

#[derive(Clone, Debug)]
struct Value<T: SmallgradFloat> {
    data: NotNan<T>,
    children: Vec<Value<T>>,
    op: ValueOp,
    grad: T,
    ident: String,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueOp {
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    None, //freshly created op
}

impl<T: SmallgradFloat> PartialEq for Value<T> {
    fn eq(&self, other: &Value<T>) -> bool {
        self.data == other.data || self.grad == other.grad || self.ident == other.ident
    }
}

impl<T: SmallgradFloat> PartialOrd for Value<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl<T: SmallgradFloat> Eq for Value<T> {}

impl<T: SmallgradFloat> Ord for Value<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

impl<T: SmallgradFloat> Hash for Value<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.data.hash(hasher)
    }
}

impl<T: SmallgradFloat> Value<T> {
    pub fn new(data: T) -> Self {
        let data = NotNan::new(data).expect("NAN/subnormal value was passed!");
        let label = random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
        Self {
            data,
            children: Vec::default(),
            op: ValueOp::None,
            grad: T::zero(),
            ident: label,
        }
    }

    pub fn update_label(&mut self, ident: &str) {
        self.ident = ident.to_string();
    }
}

impl<T: SmallgradFloat> Display for Value<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = format!("Value: {:?}", self.data);
        write!(f, "{str}")
    }
}

macro_rules! impl_num_traits_no_assign {
    ($trait: ty, $fnname: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for Value<T> {
            type Output = Value<T>;
            fn $fnname(self, other: Value<T>) -> Self {
                let mut set = Vec::new();
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                set.push(other.clone());
                set.push(self.clone());
                Self {
                    data: self.data.$fnname(other.data),
                    children: set,
                    op: ValueOp::$op_ident,
                    grad: T::zero(),
                    ident: ident,
                }
            }
        }
    };
}

macro_rules! impl_ref_ops {
    ($trait: ty, $fnname: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for &Value<T> {
            type Output = Value<T>;
            fn $fnname(self, other: &Value<T>) -> Self::Output {
                let mut vec = Vec::new();
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(self.clone());
                vec.push(other.clone());
                Value {
                    data: self.data.$fnname(other.data),
                    children: vec,
                    op: ValueOp::$op_ident,
                    grad: T::zero(),
                    ident: ident,
                }
            }
        }
    };
}

macro_rules! impl_num_traits_assign {
    ($trait: ty, $fnname: ident, $bop: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for Value<T> {
            fn $fnname(&mut self, other: Value<T>) {
                let inter = self.data.$bop(other.data);
                self.data = inter;
                let mut vec = Vec::new();
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(other.clone());
                vec.push(self.clone());
                self.children = vec;
                self.op = ValueOp::$op_ident;
                self.ident = ident;
            }
        }
    };
}

impl_num_traits_no_assign!(Add, add, Add);
impl_num_traits_no_assign!(Sub, sub, Sub);
impl_num_traits_no_assign!(Mul, mul, Mul);
impl_num_traits_no_assign!(Div, div, Div);

impl_num_traits_assign!(AddAssign, add_assign, add, Add);
impl_num_traits_assign!(SubAssign, sub_assign, sub, Sub);
impl_num_traits_assign!(MulAssign, mul_assign, mul, Mul);
impl_num_traits_assign!(DivAssign, div_assign, div, Div);

impl_ref_ops!(Add<&Value<T>>, add, Add);
impl_ref_ops!(Sub<&Value<T>>, sub, Sub);
impl_ref_ops!(Mul<&Value<T>>, mul, Mul);
impl_ref_ops!(Div<&Value<T>>, div, Div);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basic_stuff() {
        let value_a = Value::new(2f64);
        let value_b = Value::new(-3f64);
        let value_c = Value::new(10f64);
        dbg!(&value_a.children);
        dbg!(&value_b.children);
        let value_d = value_a + value_b;
        dbg!(value_d.children);
    }
}
