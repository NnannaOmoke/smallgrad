#![allow(dead_code)]

use core::fmt;

use num_traits::Float;

use rust_string_random::random;
use rust_string_random::Options;
use rust_string_random::RandWay;

use std::cell::RefCell;

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

use std::rc::Rc;

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

type DataPointer<T> = Rc<RefCell<ValueData<T>>>;

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

#[derive(Debug)]
struct Value<T: SmallgradFloat> {
    inner: DataPointer<T>,
}

impl<T: SmallgradFloat> Clone for Value<T> {
    fn clone(&self) -> Self {
        let inner_data = self.inner.borrow().clone();
        let new_cell = Rc::new(RefCell::new(inner_data));
        Self { inner: new_cell }
    }
}

impl<T: SmallgradFloat> Value<T> {
    pub fn new(data: T) -> Self {
        let inner = ValueData::new(data);
        let inner = Rc::new(RefCell::new(inner));
        Self { inner }
    }

    pub fn new_with_label(data: T, label: &str) -> Self {
        let mut inner = ValueData::new(data);
        inner.ident = label.to_string();
        let inner = Rc::new(RefCell::new(inner));
        Self { inner }
    }

    pub fn update_label(&mut self, ident: &str) {
        //we don't need a mutable, but in-case of thread safety
        (self.inner.borrow_mut()).ident = ident.to_string();
    }

    pub fn relu(&mut self) {
        //it's a little more complicated now, but we first have to clone the old state
        let prev_s = self.clone();
        let prev_s_ = prev_s.inner.clone();
        let mut bmut = self.inner.borrow_mut(); //b for borrow, just need it for convenience
        if bmut.data <= T::zero() {
            bmut.data = T::zero()
        }
        bmut.op = ValueOp::Relu;
        bmut.children = vec![prev_s_];
        bmut.ident = random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
    }

    pub fn backwards(&mut self) {
        self.inner.borrow_mut().grad = T::one();
        let children = self.toposort();
        for node in children.iter().rev() {
            node.borrow_mut().backprop()
        }
    }

    fn toposort(&mut self) -> Vec<DataPointer<T>> {
        let mut res = Vec::new();
        let mut set = HashSet::new();
        fn build_topo<T: SmallgradFloat>(
            value: DataPointer<T>,
            tset: &mut HashSet<String>,
            vec: &mut Vec<DataPointer<T>>,
        ) {
            let b = value.borrow();
            if tset.insert(b.ident.clone()) {
                for node in b.children.iter() {
                    build_topo(node.clone(), tset, vec)
                }
                vec.push(value.clone());
            }
        }
        build_topo(self.inner.clone(), &mut set, &mut res);
        res
    }
}

#[derive(Clone, Debug)]
struct ValueData<T: SmallgradFloat> {
    data: T,
    children: Vec<DataPointer<T>>,
    op: ValueOp,
    grad: T,
    ident: String,
}

impl<T: SmallgradFloat> ValueData<T> {
    fn new(data: T) -> Self {
        let label = random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
        Self {
            data,
            children: Vec::default(),
            op: ValueOp::None,
            grad: T::zero(),
            ident: label,
        }
    }
    pub fn compute_grad_wrt(&self, parent_op: ValueOp, other: Option<DataPointer<T>>) -> T {
        if let Some(other_var) = other {
            ValueOp::compute_grad(parent_op, &other_var.borrow())
        } else {
            ValueOp::compute_grad(parent_op, self)
        }
    }
    fn backprop(&mut self) {
        if self.op == ValueOp::None || self.children.is_empty() {
            return;
        }
        let mut vals = vec![];
        if self.children.len() == 2 {
            let (n1, n2) = (self.children[0].clone(), self.children[1].clone());
            vals.push(n1.borrow().compute_grad_wrt(self.op, Some(n2.clone())));
            vals.push(n2.borrow().compute_grad_wrt(self.op, Some(n1.clone())));
        } else {
            vals.push(self.children[0].borrow().compute_grad_wrt(self.op, None))
        }
        zip(vals, &self.children).for_each(|(grad, val)| {
            let total = val.borrow().grad + self.grad * grad;
            val.borrow_mut().grad = total;
        })
    }
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
    //TODO: allow for desugaring of the input: AsRef<T>
    fn compute_grad<T: SmallgradFloat>(parent: ValueOp, value_two: &ValueData<T>) -> T {
        match parent {
            Self::Add | Self::AddAssign | Self::Sub | Self::SubAssign => T::one(),
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
        self.inner.borrow().data == other.inner.borrow().data
            || self.inner.borrow().grad == other.inner.borrow().grad
            || self.inner.borrow().ident == other.inner.borrow().ident
    }
}

impl<T: SmallgradFloat> PartialOrd for Value<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner
            .borrow()
            .data
            .partial_cmp(&other.inner.borrow().data)
    }
}

impl<T: SmallgradFloat> Eq for Value<T> {}

impl<T: SmallgradFloat> Ord for Value<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.borrow().ident.cmp(&other.inner.borrow().ident)
    }
}

impl<T: SmallgradFloat> Hash for Value<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.inner.borrow().ident.hash(hasher)
    }
}

impl<T: SmallgradFloat> Display for ValueData<T> {
    //hopefully this isn't very recursive, lol, will blow the stack for larger DAG's
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = format! {"
            Value: {:?},
            Grad: {:?},
            Name: {:?}
            Children: [{:?}],
            Op: {:?}
        ", self.data, self.grad, self.ident, &self.children, self.op};
        write!(f, "{}", str)
    }
}

impl<T: SmallgradFloat> Display for Value<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner.borrow())
    }
}

macro_rules! impl_num_traits_no_assign {
    ($trait: ty, $fnname: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for Value<T> {
            type Output = Value<T>;
            fn $fnname(self, other: Value<T>) -> Self::Output {
                let mut set = Vec::new();
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                let self_value = self.inner.borrow().data;
                let other_value = other.inner.borrow().data;
                let val = self_value.$fnname(other_value);
                set.push(other.inner.clone());
                set.push(self.inner.clone());
                let inner = ValueData {
                    data: val,
                    children: set,
                    op: ValueOp::$op_ident,
                    grad: T::zero(),
                    ident: ident,
                };
                let inner = Rc::new(RefCell::new(inner));
                Self { inner: inner }
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
                let self_value = self.inner.borrow().data;
                let other_value = other.inner.borrow().data;
                let val = self_value.$fnname(other_value);
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(self.inner.clone());
                vec.push(other.inner.clone());
                let inner = ValueData {
                    data: val,
                    children: vec,
                    op: ValueOp::$op_ident,
                    grad: T::zero(),
                    ident: ident,
                };
                let inner = Rc::new(RefCell::new(inner));
                Self::Output { inner: inner }
            }
        }
    };
}
//TODO:
//find an effective way of recording this in the DAG history
macro_rules! impl_num_traits_assign {
    ($trait: ty, $fnname: ident, $bop: ident, $op_ident: ident) => {
        impl<T: SmallgradFloat> $trait for Value<T> {
            fn $fnname(&mut self, other: Value<T>) {
                let self_val = self.inner.borrow().data;
                let other_val = other.inner.borrow().data;
                //this is the old state
                let cloned = self.clone();
                let total = self_val.$bop(other_val);
                self.inner.borrow_mut().data = total;
                let mut vec = Vec::new();
                let ident =
                    random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                vec.push(other.inner.clone());
                vec.push(cloned.inner.clone()); //so basically wtf is this? //isn't this going to fuck something up?
                                                //here's what I'm thinking;
                                                //so basically if you add assign, what happens to the previous history?
                                                //we need to record it in the DAG, that's for sure
                                                //and we do so
                (self.inner.borrow_mut()).children = vec;
                (self.inner.borrow_mut()).op = ValueOp::$op_ident;
                (self.inner.borrow_mut()).ident = ident;
            }
        }
    };
}

macro_rules! impl_fops {
    ($(($fop: ident, $valueop: ident)),*) => {
            impl<T: SmallgradFloat> Value<T>{
                $(
                  pub fn $fop(&mut self){
                      let prev_state = self.clone();
                      let mut bmut = self.inner.borrow_mut();
                      let value = bmut.data.$fop();
                      bmut.data = value;
                      let ident =
                        random(9, RANDOM_STRING_CONFIG).expect("Random string initialization failed");
                      let vector = vec![prev_state.inner.clone()];
                      bmut.children = vector;
                      bmut.op = ValueOp::$valueop;
                      bmut.ident = ident;
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
        dbg!(&value_a);
        dbg!(&value_b);
        let value_d = value_a + value_b;
        dbg!(value_d);
    }

    #[test]
    fn current_state() {
        let a = Value::new(2f32);
        let b = Value::new(-3f32);
        let c = Value::new(-5f32);
        let mut d = a + b * c;
        d += Value::new(-1.5);
        d.backwards();
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
        d.backwards();
        dbg!(&d);
    }

    // #[test]
    // fn test_toposort() {
    //     let a = Value::new(56f32);
    //     let b = Value::new(8f32);
    //     let c = a * b;
    //     let mut d = c * Value::new(2f32);
    //     d.init_backprop();
    //     dbg!(&d);
    // }

    #[test]
    fn test_multigrad() {
        let a = Value::new(3f32);
        let b = a.clone() + a;
        dbg!(&b);
    }
}
