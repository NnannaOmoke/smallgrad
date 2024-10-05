#![allow(dead_code)]

use core::fmt;

use num_traits::Float;

use rust_string_random::random;
use rust_string_random::Options;
use rust_string_random::RandWay;

use std::cmp::Ordering;

use std::collections::HashSet;

use std::fmt::Debug;
use std::fmt::Display;

use std::hash::Hash;
use std::hash::Hasher;

use std::iter::zip;

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

pub trait SmallgradFloat: Debug + Float {}
impl<T: Debug + Float> SmallgradFloat for T {}

// trait Differentiable<T: SmallgradFloat> {
//     fn differentiate(&self, wrt: T) -> T;
// }

// #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
// pub enum RegisterOp<T: SmallgradFloat> {
//     FnBinary((fn(T, T) -> T, String)),
//     FnUnary((fn(T) -> T, String)),
// }

// impl<T: SmallgradFloat> Differentiable for RegisterOp<T>{
//     fn differentiate(&self, wrt: T) -> T {

//     }
// }

#[derive(Clone, Debug)]
struct Value<T: SmallgradFloat> {
    data: T,
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
    Exp,
    Sin,
    Cos,
    Tan,
    Ln,
    Sinh,
    Cosh,
    Tanh,
    Relu,
    None, //freshly created op
}

impl ValueOp {
    fn __perform_grad_compute_1d<T: SmallgradFloat>(parent: &ValueOp, value_two: &Value<T>) -> T {
        match parent {
            Self::Add | Self::AddAssign => T::one(),
            Self::Sub | Self::SubAssign => T::one(),
            Self::Mul | Self::MulAssign => value_two.data,
            Self::Div | Self::DivAssign => T::one() / value_two.data,
            Self::Exp => T::exp(value_two.data),
            Self::Sin => T::cos(value_two.data),
            Self::Cos => T::sin(value_two.data.neg()),
            Self::Tan => T::one() / T::cos(value_two.data).powi(2),
            Self::Ln => T::one() / value_two.data,
            Self::Sinh => T::cosh(value_two.data),
            Self::Cosh => T::sinh(value_two.data),
            Self::Tanh => T::one() / T::cosh(value_two.data).powi(2),
            Self::Relu => {
                if value_two.data > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            _ => T::zero(),
        }
    }
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
        self.ident.cmp(&other.ident)
    }
}

impl<T: SmallgradFloat> Hash for Value<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.ident.hash(hasher)
    }
}

impl<T: SmallgradFloat> Value<T> {
    pub fn new(data: T) -> Self {
        let label = random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
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

    pub fn init_backprop_recursive(&mut self) {
        self.grad = T::one();
        self.__backprop();
    }

    pub fn relu(&mut self) {
        if self.data > T::zero() {
        } else {
            self.data = T::zero()
        };
        self.op = ValueOp::Relu;
        self.children = vec![self.clone()];
        self.ident = random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
    }

    fn __compute_grad_wrt(&self, parent_op: &ValueOp, other: Option<&Value<T>>) -> T {
        match other {
            Some(other) => {
                //we have a binary operation
                ValueOp::__perform_grad_compute_1d(parent_op, other)
            }
            None => {
                //we have a unary operation
                //the other variable passed to __perform_grad_compute_1d() will be the self.data
                ValueOp::__perform_grad_compute_1d(parent_op, self)
            }
        }
    }

    fn __backprop(&mut self) {
        if self.op == ValueOp::None || self.children.len() == 0 {
            return;
        }
        let mut val = vec![];
        if self.children.len() == 2 {
            let (node_one, node_two) = (&self.children[0], &self.children[1]);
            val.push(node_one.__compute_grad_wrt(&self.op, Some(node_two)));
            val.push(node_two.__compute_grad_wrt(&self.op, Some(node_one)));
        } else {
            val.push(self.children[0].__compute_grad_wrt(&self.op, None));
        }
        zip(val, &mut self.children).for_each(|(grad, val)| val.grad = self.grad * grad);
        for child in self.children.iter_mut() {
            child.__backprop();
        }
    }

    fn __backprop_non_recursive(&mut self) {
        if self.op == ValueOp::None || self.children.len() == 0 {
            return;
        }
        let mut val = vec![];
        if self.children.len() == 2 {
            let (node_one, node_two) = (&self.children[0], &self.children[1]);
            val.push(node_one.__compute_grad_wrt(&self.op, Some(node_two)));
            val.push(node_two.__compute_grad_wrt(&self.op, Some(node_one)));
        } else {
            val.push(self.children[0].__compute_grad_wrt(&self.op, None));
        }
        zip(val, &mut self.children).for_each(|(grad, val)| val.grad = self.grad * grad);
    }

    pub fn init_backprop(&mut self) {
        self.grad = T::one();
        let mut kids = unsafe { self.__toposort() };
        for kid in kids.iter_mut().rev() {
            (*kid).__backprop();
        }
    }

    //this is unsafe cheese, please don't let this devalue my worth as a programmer :(
    unsafe fn __toposort(&mut self) -> Vec<&mut Value<T>> {
        let mut res = Vec::new();
        let mut setcontain = HashSet::new();
        unsafe fn __build_topo<'lt, T: SmallgradFloat>(
            value: *mut Value<T>,
            setcontain: &mut HashSet<String>,
            vec: &mut Vec<*mut Value<T>>,
        ) {
            if setcontain.insert((*value).ident.clone()) {
                for child in (*value).children.iter_mut() {
                    let child = child as *mut Value<T>;
                    __build_topo(child, setcontain, vec);
                }
                vec.push(value);
            }
        }
        __build_topo(self, &mut setcontain, &mut res);
        res.iter_mut()
            .map(|e| e.as_mut().unwrap())
            .collect::<Vec<_>>()
    }

    fn __traverse_dag(&mut self) {
        let mut tracker = Vec::new();
        let mut result = Vec::new();
        let mut discovered = HashSet::new();
        discovered.insert(self.ident.clone());
        tracker.push(self);
        loop {
            if tracker.is_empty() {
                break;
            }
            result.push(tracker.pop())
        }
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
                    random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                let val = self.data.$fnname(other.data);
                set.push(other.clone());
                set.push(self);
                Self {
                    data: val,
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
                let val = self.data.$fnname(other.data);
                let ident =
                    random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(self.clone());
                vec.push(other.clone());
                Value {
                    data: val,
                    children: vec,
                    op: ValueOp::$op_ident,
                    grad: T::zero(),
                    ident: ident,
                }
            }
        }
    };
}
//find an effective way of recording this in the DAG history
macro_rules! impl_num_traits_assign {
    ($trait: ty, $fnname: ident, $bop: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for Value<T> {
            fn $fnname(&mut self, other: Value<T>) {
                let inter = self.data.$bop(other.data);
                self.data = inter;
                let mut vec = Vec::new();
                let ident =
                    random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(other.clone());
                vec.push(self.clone());
                self.children = vec;
                self.op = ValueOp::$op_ident;
                self.ident = ident;
            }
        }
    };
}

macro_rules! impl_fops {
    ($(($fop: ident, $valueop: ident)),*) => {
            impl<T: SmallgradFloat> Value<T>{
                $(
                  pub fn $fop(&mut self){
                      self.data = self.data.$fop();
                      let ident =
                        random(5, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                      let vector = vec![self.clone()];
                      self.children = vector;
                      self.op = ValueOp::$valueop;
                      self.ident = ident;
                  }
                )*
            }

    };
}

impl_num_traits_no_assign!(Add, add, Add);
impl_num_traits_no_assign!(Sub, sub, Sub);
impl_num_traits_no_assign!(Mul, mul, Mul);
impl_num_traits_no_assign!(Div, div, Div);

impl_num_traits_assign!(AddAssign, add_assign, add, AddAssign);
impl_num_traits_assign!(SubAssign, sub_assign, sub, SubAssign);
impl_num_traits_assign!(MulAssign, mul_assign, mul, MulAssign);
impl_num_traits_assign!(DivAssign, div_assign, div, DivAssign);

impl_ref_ops!(Add<&Value<T>>, add, Add);
impl_ref_ops!(Sub<&Value<T>>, sub, Sub);
impl_ref_ops!(Mul<&Value<T>>, mul, Mul);
impl_ref_ops!(Div<&Value<T>>, div, Div);

impl_fops!(
    (exp, Exp),
    (sin, Sin),
    (cos, Cos),
    (tan, Tan),
    (ln, Ln),
    (sinh, Sinh),
    (cosh, Cosh),
    (tanh, Tanh)
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basic_stuff() {
        let value_a = Value::new(2f64);
        let value_b = Value::new(-3f64);
        dbg!(&value_a.children);
        dbg!(&value_b.children);
        let value_d = value_a + value_b;
        dbg!(value_d.children);
    }

    #[test]
    fn current_state() {
        let a = Value::new(2f32);
        let b = Value::new(-3f32);
        let c = Value::new(-5f32);
        let mut d = a + b * c;
        d += Value::new(-1.5);
        d.init_backprop();
        dbg!(&d);
    }
    #[test]
    fn test_added_fns() {
        let mut a = Value::new(2f32);
        a.update_label("a");
        let mut b = Value::new(-3f32);
        b.update_label("b");
        let mut c = Value::new(-5f32);
        c.update_label("c");
        a.sin();
        let mut inter = a + b;
        inter.update_label("inter");
        inter.sin();
        let mut d = inter / c;
        d += Value::new(-1.5);
        d.init_backprop();
        dbg!(&d);
    }

    #[test]
    fn test_toposort() {
        let a = Value::new(56f32);
        let b = Value::new(8f32);
        let c = a * b;
        let mut d = c * Value::new(2f32);
        d.init_backprop();
        dbg!(&d);
    }
}
