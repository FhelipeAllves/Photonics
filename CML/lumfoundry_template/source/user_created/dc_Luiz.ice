      type        INTERCONNECT_PRIMITIVE_ELEMENT    version        8    �������� ���              c o n n e c t i o n s   	               p i n s   	         x m l   o b j e c t            i d   
     0    s t a r t   n a m e   
    0 : : d c _ L u i z : : R E L A Y _ 3 : : p o r t    r o u t i n g            e n d   n a m e   
    2 : : d c _ L u i z : : S P A R _ 1 : : p o r t   1           p i n s   	         x m l   o b j e c t            i d   
     0    s t a r t   n a m e   
    0 : : d c _ L u i z : : R E L A Y _ 1 : : p o r t    r o u t i n g            e n d   n a m e   
    2 : : d c _ L u i z : : S P A R _ 1 : : p o r t   3           p i n s   	         x m l   o b j e c t            i d   
     0    s t a r t   n a m e   
    0 : : d c _ L u i z : : R E L A Y _ 2 : : p o r t    r o u t i n g            e n d   n a m e   
    2 : : d c _ L u i z : : S P A R _ 1 : : p o r t   4           p i n s   	         x m l   o b j e c t            i d   
     0    s t a r t   n a m e   
    2 : : d c _ L u i z : : S P A R _ 1 : : p o r t   2    r o u t i n g            e n d   n a m e   
    0 : : d c _ L u i z : : R E L A Y _ 4 : : p o r t    p r o p e r t i e s          " s t a t i c   p r o p e r t i e s       ;   $ n u m b e r   o f   f i r   t a p s           d i a g n o s t i c   s i z e          $ f i l t e r   f i t   r o l l o f f    ?�������    e x p a n d a b l e        f i l t e r   d e l a y               ( f i l t e r   f i t   t o l e r a n c e    ?�������    y   p o s i t i o n    @b�         a n a l y s i s   s c r i p t   
 ����    w i n d o w   f u n c t i o n       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                       c h o i c e s           r e c t a n g u l a r    h a m m i n g    h a n n i n g    a n n o t a t e       
 m o d e s   
     T E   > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s       2   . c h e c k   i n t e r n a l   m o n i t o r s        d e l a y   i n s e r t i o n       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                    c h o i c e s           o u t p u t   p o r t s   & b i d i r e c t i o n a l   p o r t s    m c s   f i l e n a m e   
 ����     r u n   s e t u p   s c r i p t       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                    c h o i c e s           a u t o m a t i c    a l w a y s   $ o u t p u t   s i g n a l   m o d e       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                    c h o i c e s           s a m p l e   
 b l o c k    b i t r a t e    BHv�      , i n i t i a l i z e   f i l t e r   t a p s         l a y o u t   n a m e   
 ����    m c s   
 ����    l o c a l   p a t h   
 ����    i m p o r t   f i l e   
 ����   0 d i a g n o s t i c   d a t a   f i l e n a m e   
     s p a r a m . d a t    d e s c r i p t i o n   
    ~ C o m p o u n d   i s   a   h i e r a r c h i c a l   e l e m e n t   t h a t   c o n t a i n s   o t h e r   e l e m e n t s    s a m p l e   r a t e    BwHv�      < i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t       & d e a d l o c k   r e s o l u t i o n       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                    c h o i c e s           i g n o r e    p r e v e n t    t y p e   
      C o m p o u n d   E l e m e n t    t i m e   w i n d o w    >5��yd�    e n a b l e d        v e r s i o n    ����   , d i a g n o s t i c   d a t a   e x p o r t         p r e f i x   
     C O M P O U N D    t e m p e r a t u r e    @r�        $ h o r i z o n t a l   f l i p p e d        4 m a x i m u m   n u m b e r   o f   i i r   t a p s          4 d i a g n o s t i c   s t a r t   f r e q u e n c y    B��~���     s p t   f i l e   
 ����    c o m p o n e n t   i d       0 n u m b e r   o f   o u t p u t   s i g n a l s          2 n u m b e r   o f   t a p s   e s t i m a t i o n       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                       c h o i c e s           d i s a b l e d    f i t   t o l e r a n c e    g r o u p   d e l a y   
 m o d e l   
 ����    x   p o s i t i o n    �P�          v e r t i c a l   f l i p p e d        2 d i a g n o s t i c   s t o p   f r e q u e n c y    B��g�m�     k e y w o r d s   
 ����    l i b r a r y   
 ����   0 s c a t t e r i n g   d a t a   a n a l y s i s         x   c o o r d i n a t e       lumatrix__matrixd        ����                                 n a m e   
     d c _ L u i z    s e t u p   s c r i p t   
   � f i l e n a m e = " C : / U s e r s / a l v e s / D o c u m e n t s / P h o t o n i c s / C M L / d c _ m a p . x m l " ; 
 t a b l e   =   " d i r e c t i o n a l _ c o u p l e r " ; 
 
 d e s i g n   =   c e l l ( 1 ) ; 
 d e s i g n { 1 }   =   s t r u c t ; 
 d e s i g n { 1 } . n a m e   =   " L c " ; 
 d e s i g n { 1 } . v a l u e   =   c o u p l i n g _ l e n g t h ; 
 
 s e t n a m e d ( ' S P A R _ 1 ' , ' l o a d   f r o m   f i l e ' , 0 ) ; 
 M   =   l o o k u p r e a d n p o r t s p a r a m e t e r (   f i l e n a m e ,   t a b l e ,   d e s i g n ,   " s - p a r a m "   ) ; 
 s e t v a l u e ( ' S P A R _ 1 ' , ' s   p a r a m e t e r s ' , M ) ; 
   $ n u m b e r   o f   i i r   t a p s          & d i g i t a l   f i l t e r   t y p e       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                       c h o i c e s           s i n g l e   t a p    F I R    I I R    r o t a t e d            r u n   d i a g n o s t i c         y   c o o r d i n a t e       lumatrix__matrixd        ����                                 u r l   
 ����   4 m a x i m u m   n u m b e r   o f   f i r   t a p s          $ d y n a m i c   p r o p e r t i e s           c o u p l i n g _ l e n g t h            m e t a   d a t a       
    o p t i o n s       <   $ n u m b e r   o f   f i r   t a p s           d i a g n o s t i c   s i z e          $ f i l t e r   f i t   r o l l o f f           e x p a n d a b l e            f i l t e r   d e l a y          ( f i l t e r   f i t   t o l e r a n c e           y   p o s i t i o n           a n a l y s i s   s c r i p t           w i n d o w   f u n c t i o n           a n n o t a t e           
 m o d e s          > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s          . c h e c k   i n t e r n a l   m o n i t o r s            d e l a y   i n s e r t i o n            m c s   f i l e n a m e             r u n   s e t u p   s c r i p t           $ o u t p u t   s i g n a l   m o d e            b i t r a t e           , i n i t i a l i z e   f i l t e r   t a p s           l a y o u t   n a m e           m c s            l o c a l   p a t h           i m p o r t   f i l e          0 d i a g n o s t i c   d a t a   f i l e n a m e           d e s c r i p t i o n            s a m p l e   r a t e           < i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t           & d e a d l o c k   r e s o l u t i o n            c o u p l i n g _ l e n g t h            t y p e           t i m e   w i n d o w            e n a b l e d            v e r s i o n           , d i a g n o s t i c   d a t a   e x p o r t           p r e f i x            t e m p e r a t u r e           $ h o r i z o n t a l   f l i p p e d          4 m a x i m u m   n u m b e r   o f   i i r   t a p s          4 d i a g n o s t i c   s t a r t   f r e q u e n c y           s p t   f i l e           c o m p o n e n t   i d           0 n u m b e r   o f   o u t p u t   s i g n a l s          2 n u m b e r   o f   t a p s   e s t i m a t i o n          
 m o d e l            x   p o s i t i o n            v e r t i c a l   f l i p p e d          2 d i a g n o s t i c   s t o p   f r e q u e n c y           k e y w o r d s           l i b r a r y           0 s c a t t e r i n g   d a t a   a n a l y s i s            x   c o o r d i n a t e           n a m e            s e t u p   s c r i p t          $ n u m b e r   o f   i i r   t a p s          & d i g i t a l   f i l t e r   t y p e           r o t a t e d           r u n   d i a g n o s t i c           y   c o o r d i n a t e           u r l           4 m a x i m u m   n u m b e r   o f   f i r   t a p s          
 k i n d s       <   $ n u m b e r   o f   f i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    d i a g n o s t i c   s i z e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   $ f i l t e r   f i t   r o l l o f f       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    e x p a n d a b l e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    f i l t e r   d e l a y       LumQuantityKind        u n i t   
     s    s t a n d a r d   u n i t   
     s    k i n d   
     T i m e   ( f i l t e r   f i t   t o l e r a n c e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n a l y s i s   s c r i p t       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    w i n d o w   f u n c t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   
 m o d e s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   . c h e c k   i n t e r n a l   m o n i t o r s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e l a y   i n s e r t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    m c s   f i l e n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     r u n   s e t u p   s c r i p t       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ o u t p u t   s i g n a l   m o d e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    b i t r a t e       LumQuantityKind        u n i t   
     b i t s / s    s t a n d a r d   u n i t   
     b i t s / s    k i n d   
     B i t r a t e   , i n i t i a l i z e   f i l t e r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l a y o u t   n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    m c s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    i m p o r t   f i l e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   0 d i a g n o s t i c   d a t a   f i l e n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    s a m p l e   r a t e       LumQuantityKind        u n i t   
     H z    s t a n d a r d   u n i t   
     H z    k i n d   
     F r e q u e n c y   < i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   & d e a d l o c k   r e s o l u t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    c o u p l i n g _ l e n g t h       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     D i s t a n c e    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t i m e   w i n d o w       LumQuantityKind        u n i t   
     s    s t a n d a r d   u n i t   
     s    k i n d   
     T i m e    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    v e r s i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   , d i a g n o s t i c   d a t a   e x p o r t       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t e m p e r a t u r e       LumQuantityKind        u n i t   
     K    s t a n d a r d   u n i t   
     K    k i n d   
     T e m p e r a t u r e   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   4 m a x i m u m   n u m b e r   o f   i i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   4 d i a g n o s t i c   s t a r t   f r e q u e n c y       LumQuantityKind        u n i t   
     T H z    s t a n d a r d   u n i t   
     H z    k i n d   
     F r e q u e n c y    s p t   f i l e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    c o m p o n e n t   i d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   0 n u m b e r   o f   o u t p u t   s i g n a l s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   2 n u m b e r   o f   t a p s   e s t i m a t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   2 d i a g n o s t i c   s t o p   f r e q u e n c y       LumQuantityKind        u n i t   
     T H z    s t a n d a r d   u n i t   
     H z    k i n d   
     F r e q u e n c y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   0 s c a t t e r i n g   d a t a   a n a l y s i s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    s e t u p   s c r i p t       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ n u m b e r   o f   i i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   & d i g i t a l   f i l t e r   t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r u n   d i a g n o s t i c       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   4 m a x i m u m   n u m b e r   o f   f i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    p r i o r i t i e s       <   $ n u m b e r   o f   f i r   t a p s      �    d i a g n o s t i c   s i z e      �   $ f i l t e r   f i t   r o l l o f f      �    e x p a n d a b l e       �    f i l t e r   d e l a y      �   ( f i l t e r   f i t   t o l e r a n c e      �    y   p o s i t i o n       �    a n a l y s i s   s c r i p t      h    w i n d o w   f u n c t i o n      �    a n n o t a t e       2   
 m o d e s      T   > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s      �   . c h e c k   i n t e r n a l   m o n i t o r s      @    d e l a y   i n s e r t i o n      6    m c s   f i l e n a m e      D     r u n   s e t u p   s c r i p t      "   $ o u t p u t   s i g n a l   m o d e          b i t r a t e       �   , i n i t i a l i z e   f i l t e r   t a p s      �    l a y o u t   n a m e      b    m c s      N    l o c a l   p a t h       x    i m p o r t   f i l e      l   0 d i a g n o s t i c   d a t a   f i l e n a m e          d e s c r i p t i o n       P    s a m p l e   r a t e       �   < i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t      0   & d e a d l o c k   r e s o l u t i o n      ,    c o u p l i n g _ l e n g t h      v    t y p e       F    t i m e   w i n d o w       �    e n a b l e d       <    v e r s i o n      &   , d i a g n o s t i c   d a t a   e x p o r t      �    p r e f i x       Z    t e m p e r a t u r e         $ h o r i z o n t a l   f l i p p e d       �   4 m a x i m u m   n u m b e r   o f   i i r   t a p s      �   4 d i a g n o s t i c   s t a r t   f r e q u e n c y          s p t   f i l e      X    c o m p o n e n t   i d      :   0 n u m b e r   o f   o u t p u t   s i g n a l s         2 n u m b e r   o f   t a p s   e s t i m a t i o n      |   
 m o d e l       d    x   p o s i t i o n       �     v e r t i c a l   f l i p p e d       �   2 d i a g n o s t i c   s t o p   f r e q u e n c y          k e y w o r d s       �    l i b r a r y       n   0 s c a t t e r i n g   d a t a   a n a l y s i s      J    x   c o o r d i n a t e       �    n a m e       (    s e t u p   s c r i p t      ^   $ n u m b e r   o f   i i r   t a p s      �   & d i g i t a l   f i l t e r   t y p e      r    r o t a t e d       �    r u n   d i a g n o s t i c      �    y   c o o r d i n a t e       �    u r l       �   4 m a x i m u m   n u m b e r   o f   f i r   t a p s      �    c a t e g o r i e s           c o u p l i n g _ l e n g t h   
 ����   
 t y p e s           c o u p l i n g _ l e n g t h            e x p r e s s i o n s       <   $ n u m b e r   o f   f i r   t a p s   
         d i a g n o s t i c   s i z e   
        $ f i l t e r   f i t   r o l l o f f   
         e x p a n d a b l e   
         f i l t e r   d e l a y   
        ( f i l t e r   f i t   t o l e r a n c e   
         y   p o s i t i o n   
         a n a l y s i s   s c r i p t   
         w i n d o w   f u n c t i o n   
         a n n o t a t e   
        
 m o d e s   
        > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s   
        . c h e c k   i n t e r n a l   m o n i t o r s   
         d e l a y   i n s e r t i o n   
    " % d e l a y   i n s e r t i o n %    m c s   f i l e n a m e   
          r u n   s e t u p   s c r i p t   
        $ o u t p u t   s i g n a l   m o d e   
    ( % o u t p u t   s i g n a l   m o d e %    b i t r a t e   
     % b i t r a t e %   , i n i t i a l i z e   f i l t e r   t a p s   
         l a y o u t   n a m e   
         m c s   
         l o c a l   p a t h   
         i m p o r t   f i l e   
        0 d i a g n o s t i c   d a t a   f i l e n a m e   
         d e s c r i p t i o n   
         s a m p l e   r a t e   
     % s a m p l e   r a t e %   < i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t   
    @ % i n t e r n a l   e l e c t r i c a l   e q u i v a l e n t %   & d e a d l o c k   r e s o l u t i o n   
    * % d e a d l o c k   r e s o l u t i o n %    c o u p l i n g _ l e n g t h   
 ����    t y p e   
         t i m e   w i n d o w   
     % t i m e   w i n d o w %    e n a b l e d   
         v e r s i o n   
        , d i a g n o s t i c   d a t a   e x p o r t   
         p r e f i x   
         t e m p e r a t u r e   
     % t e m p e r a t u r e %   $ h o r i z o n t a l   f l i p p e d   
        4 m a x i m u m   n u m b e r   o f   i i r   t a p s   
        4 d i a g n o s t i c   s t a r t   f r e q u e n c y   
         s p t   f i l e   
         c o m p o n e n t   i d   
        0 n u m b e r   o f   o u t p u t   s i g n a l s   
    4 % n u m b e r   o f   o u t p u t   s i g n a l s %   2 n u m b e r   o f   t a p s   e s t i m a t i o n   
        
 m o d e l   
         x   p o s i t i o n   
          v e r t i c a l   f l i p p e d   
        2 d i a g n o s t i c   s t o p   f r e q u e n c y   
         k e y w o r d s   
         l i b r a r y   
        0 s c a t t e r i n g   d a t a   a n a l y s i s   
         x   c o o r d i n a t e   
         n a m e   
         s e t u p   s c r i p t   
        $ n u m b e r   o f   i i r   t a p s   
        & d i g i t a l   f i l t e r   t y p e   
         r o t a t e d   
         r u n   d i a g n o s t i c   
         y   c o o r d i n a t e   
         u r l   
        4 m a x i m u m   n u m b e r   o f   f i r   t a p s   
         m e t a   d a t a           c o u p l i n g _ l e n g t h   
 ����    l i m i t s           c o u p l i n g _ l e n g t h       	LumLimit        l o w e r   l i m i t    ԲI�%��}    t y p e       
    u p p e r   l i m i t    T�I�%��}    d e p e n d e n c i e s           c o u p l i n g _ l e n g t h   
 ����    a l l   c a t e g o r i e s       ����   0 N u m e r i c a l / D i g i t a l   F i l t e r    S c r i p t   " D e s i g n   K i t / H e a d e r    S i m u l a t i o n    T h e r m a l    G e n e r a l    V a l i d a t i o n    D i a g n o s t i c    D e s i g n   K i t    N u m e r i c a l   $ p r i v a t e   p r o p e r t i e s          0 a n a l y s i s   s c r i p t   i s   s t a l e        i c o n      L~<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="64"
   height="64"
   id="svg3110"
   version="1.1"
   inkscape:version="0.48.5 r10040"
   sodipodi:docname="lcs_sphere_bg.svg">
  <defs
     id="defs3112">
    <linearGradient
       id="linearGradient5710">
      <stop
         id="stop5712"
         offset="0"
         style="stop-color:#000000;stop-opacity:0.71681416" />
      <stop
         style="stop-color:#2b2b2b;stop-opacity:0.47345133"
         offset="0.23969014"
         id="stop5714" />
      <stop
         style="stop-color:#396f7c;stop-opacity:0.69433957;"
         offset="0.45722306"
         id="stop5716" />
      <stop
         style="stop-color:#ffffff;stop-opacity:0.15425532"
         offset="0.65027237"
         id="stop5718" />
      <stop
         id="stop5720"
         offset="1"
         style="stop-color:#ffffff;stop-opacity:0" />
    </linearGradient>
    <linearGradient
       id="linearGradient4085">
      <stop
         style="stop-color:#000000;stop-opacity:0.20377359;"
         offset="0"
         id="stop4087" />
      <stop
         id="stop4089"
         offset="0.31692868"
         style="stop-color:#2b2b2b;stop-opacity:0.59622639;" />
      <stop
         id="stop4091"
         offset="0.53031564"
         style="stop-color:#396f7c;stop-opacity:0.69433957;" />
      <stop
         id="stop4093"
         offset="0.83225203"
         style="stop-color:#393939;stop-opacity:0.32452831;" />
      <stop
         style="stop-color:#000000;stop-opacity:0.21132076;"
         offset="1"
         id="stop4095" />
    </linearGradient>
    <linearGradient
       id="linearGradient4016">
      <stop
         id="stop4018"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024" />
      <stop
         id="stop4026"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 32 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="64 : 32 : 1"
       inkscape:persp3d-origin="32 : 21.333333 : 1"
       id="perspective3963" />
    <linearGradient
       id="linearGradient3937">
      <stop
         style="stop-color:#ffffff;stop-opacity:1;"
         offset="0"
         id="stop3957" />
      <stop
         id="stop3965"
         offset="0.23357147"
         style="stop-color:#dae4e6;stop-opacity:1;" />
      <stop
         id="stop3961"
         offset="0.50995117"
         style="stop-color:#557380;stop-opacity:1;" />
      <stop
         id="stop3959"
         offset="0.74842322"
         style="stop-color:#393939;stop-opacity:1;" />
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="1"
         id="stop3941" />
    </linearGradient>
    <radialGradient
       inkscape:collect="always"
       xlink:href="#linearGradient3937-8"
       id="radialGradient3953-8"
       cx="41.800537"
       cy="47.421951"
       fx="41.800537"
       fy="47.421951"
       r="12.216533"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(-1.4950146,-1.289968,1.2681304,-1.4697057,51.555346,148.68724)" />
    <linearGradient
       id="linearGradient3937-8">
      <stop
         style="stop-color:#ffffff;stop-opacity:1;"
         offset="0"
         id="stop3957-2" />
      <stop
         id="stop3965-7"
         offset="0.31492463"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         id="stop3961-7"
         offset="0.50995117"
         style="stop-color:#80555b;stop-opacity:1;" />
      <stop
         id="stop3959-1"
         offset="0.74842322"
         style="stop-color:#393939;stop-opacity:1;" />
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="1"
         id="stop3941-2" />
    </linearGradient>
    <radialGradient
       r="12.216533"
       fy="47.338062"
       fx="41.706768"
       cy="47.338062"
       cx="41.706768"
       gradientTransform="matrix(-1.9089582,-1.2639863,1.2387528,-1.8708484,70.259621,166.51498)"
       gradientUnits="userSpaceOnUse"
       id="radialGradient3991-0"
       xlink:href="#linearGradient4016-4"
       inkscape:collect="always" />
    <linearGradient
       id="linearGradient4016-4">
      <stop
         id="stop4018-9"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020-4" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022-6" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024-0" />
      <stop
         id="stop4026-3"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <radialGradient
       r="12.216533"
       fy="47.338062"
       fx="41.706768"
       cy="47.338062"
       cx="41.706768"
       gradientTransform="matrix(-1.9089582,-1.2639863,1.2387528,-1.8708484,70.259621,166.51498)"
       gradientUnits="userSpaceOnUse"
       id="radialGradient3991-2"
       xlink:href="#linearGradient4016-45"
       inkscape:collect="always" />
    <linearGradient
       id="linearGradient4016-45">
      <stop
         id="stop4018-8"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020-8" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022-1" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024-2" />
      <stop
         id="stop4026-1"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient4085"
       id="linearGradient5604"
       x1="38.899879"
       y1="39.905556"
       x2="46.964371"
       y2="31.924204"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(0.75938469,0,0,0.75938469,31.258584,1.43463)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient4085-8"
       id="linearGradient5604-9"
       x1="38.899879"
       y1="39.905556"
       x2="46.964371"
       y2="31.924204"
       gradientUnits="userSpaceOnUse" />
    <linearGradient
       id="linearGradient4085-8">
      <stop
         style="stop-color:#000000;stop-opacity:0.57358491;"
         offset="0"
         id="stop4087-2" />
      <stop
         id="stop4089-4"
         offset="0.31692868"
         style="stop-color:#2b2b2b;stop-opacity:0.59622639;" />
      <stop
         id="stop4091-5"
         offset="0.53031564"
         style="stop-color:#396f7c;stop-opacity:0.69433957;" />
      <stop
         id="stop4093-2"
         offset="0.83225203"
         style="stop-color:#393939;stop-opacity:0.32452831;" />
      <stop
         style="stop-color:#000000;stop-opacity:0.21132076;"
         offset="1"
         id="stop4095-9" />
    </linearGradient>
    <linearGradient
       y2="31.924204"
       x2="46.964371"
       y1="39.905556"
       x1="38.899879"
       gradientUnits="userSpaceOnUse"
       id="linearGradient5652"
       xlink:href="#linearGradient4085-8"
       inkscape:collect="always" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="12.515625"
     inkscape:cx="19.535581"
     inkscape:cy="32"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="false"
     inkscape:window-width="1920"
     inkscape:window-height="1018"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata3115">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-988.36218)">
    <image
       y="990.56219"
       x="0"
       id="image3046"
       xlink:href="data:image/png;base64,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 YII= "
       height="60"
       width="64"
       style="opacity:0.15" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880"
       width="4"
       height="4"
       x="0"
       y="1002.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1"
       width="4"
       height="4"
       x="0"
       y="1034.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1-7"
       width="4"
       height="4"
       x="60"
       y="1002.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1-7-4"
       width="4"
       height="4"
       x="60"
       y="1034.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-0"
       width="24.049999"
       height="4"
       x="20"
       y="1014.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-0-9"
       width="24.049999"
       height="4"
       x="20"
       y="1022.3622" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 20.190939,1016.3912 c -2.727844,-0.2332 -4.854091,-1.6454 -5.386257,-2.463 -0.600298,-0.9223 -1.825207,-3.1963 -2.887612,-4.8923 -1.053476,-1.6818 -2.2256099,-3.3725 -3.0390361,-3.962 -0.7246268,-0.5252 -2.4672914,-0.7403 -4.9280952,-0.7085"
       id="path3909"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 60.241352,1004.3654 c -2.727903,0.2329 -4.854196,1.6447 -5.386374,2.4624 -0.600311,0.9221 -1.825246,3.1954 -2.887674,4.8912 -1.053499,1.6814 -2.225658,3.3715 -3.039102,3.9608 -0.724642,0.5252 -2.467345,0.7404 -4.928202,0.7083"
       id="path3909-4"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 60.269449,1036.3622 c -2.727844,-0.2332 -4.854091,-1.645 -5.386257,-2.4625 -0.600298,-0.9222 -1.825207,-3.1958 -2.887612,-4.8914 -1.053476,-1.6815 -2.225609,-3.3717 -3.039036,-3.9611 -0.724626,-0.5252 -2.467291,-0.7404 -4.928095,-0.7084"
       id="path3909-8"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 20.191291,1024.3359 c -2.727903,0.2328 -4.854196,1.6447 -5.386374,2.4623 -0.600311,0.9223 -1.825246,3.1957 -2.887674,4.8915 -1.053499,1.6815 -2.2256583,3.3717 -3.0391025,3.9611 -0.724642,0.5251 -2.467345,0.7404 -4.9282018,0.7083"
       id="path3909-82"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
  </g>
</svg>
    v i e w   t r a n s f o r m   Q ?�                              ?�                              ?�          v a r i a b l e   m a p           
 v a l i d        b o u n d i n g   r e c t                    @P      @P         * s e t u p   s c r i p t   i s   s t a l e         d a r k   i c o n      L~<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="64"
   height="64"
   id="svg3110"
   version="1.1"
   inkscape:version="0.48.5 r10040"
   sodipodi:docname="lcs_sphere_bg.svg">
  <defs
     id="defs3112">
    <linearGradient
       id="linearGradient5710">
      <stop
         id="stop5712"
         offset="0"
         style="stop-color:#000000;stop-opacity:0.71681416" />
      <stop
         style="stop-color:#2b2b2b;stop-opacity:0.47345133"
         offset="0.23969014"
         id="stop5714" />
      <stop
         style="stop-color:#396f7c;stop-opacity:0.69433957;"
         offset="0.45722306"
         id="stop5716" />
      <stop
         style="stop-color:#ffffff;stop-opacity:0.15425532"
         offset="0.65027237"
         id="stop5718" />
      <stop
         id="stop5720"
         offset="1"
         style="stop-color:#ffffff;stop-opacity:0" />
    </linearGradient>
    <linearGradient
       id="linearGradient4085">
      <stop
         style="stop-color:#000000;stop-opacity:0.20377359;"
         offset="0"
         id="stop4087" />
      <stop
         id="stop4089"
         offset="0.31692868"
         style="stop-color:#2b2b2b;stop-opacity:0.59622639;" />
      <stop
         id="stop4091"
         offset="0.53031564"
         style="stop-color:#396f7c;stop-opacity:0.69433957;" />
      <stop
         id="stop4093"
         offset="0.83225203"
         style="stop-color:#393939;stop-opacity:0.32452831;" />
      <stop
         style="stop-color:#000000;stop-opacity:0.21132076;"
         offset="1"
         id="stop4095" />
    </linearGradient>
    <linearGradient
       id="linearGradient4016">
      <stop
         id="stop4018"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024" />
      <stop
         id="stop4026"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 32 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="64 : 32 : 1"
       inkscape:persp3d-origin="32 : 21.333333 : 1"
       id="perspective3963" />
    <linearGradient
       id="linearGradient3937">
      <stop
         style="stop-color:#ffffff;stop-opacity:1;"
         offset="0"
         id="stop3957" />
      <stop
         id="stop3965"
         offset="0.23357147"
         style="stop-color:#dae4e6;stop-opacity:1;" />
      <stop
         id="stop3961"
         offset="0.50995117"
         style="stop-color:#557380;stop-opacity:1;" />
      <stop
         id="stop3959"
         offset="0.74842322"
         style="stop-color:#393939;stop-opacity:1;" />
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="1"
         id="stop3941" />
    </linearGradient>
    <radialGradient
       inkscape:collect="always"
       xlink:href="#linearGradient3937-8"
       id="radialGradient3953-8"
       cx="41.800537"
       cy="47.421951"
       fx="41.800537"
       fy="47.421951"
       r="12.216533"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(-1.4950146,-1.289968,1.2681304,-1.4697057,51.555346,148.68724)" />
    <linearGradient
       id="linearGradient3937-8">
      <stop
         style="stop-color:#ffffff;stop-opacity:1;"
         offset="0"
         id="stop3957-2" />
      <stop
         id="stop3965-7"
         offset="0.31492463"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         id="stop3961-7"
         offset="0.50995117"
         style="stop-color:#80555b;stop-opacity:1;" />
      <stop
         id="stop3959-1"
         offset="0.74842322"
         style="stop-color:#393939;stop-opacity:1;" />
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="1"
         id="stop3941-2" />
    </linearGradient>
    <radialGradient
       r="12.216533"
       fy="47.338062"
       fx="41.706768"
       cy="47.338062"
       cx="41.706768"
       gradientTransform="matrix(-1.9089582,-1.2639863,1.2387528,-1.8708484,70.259621,166.51498)"
       gradientUnits="userSpaceOnUse"
       id="radialGradient3991-0"
       xlink:href="#linearGradient4016-4"
       inkscape:collect="always" />
    <linearGradient
       id="linearGradient4016-4">
      <stop
         id="stop4018-9"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020-4" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022-6" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024-0" />
      <stop
         id="stop4026-3"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <radialGradient
       r="12.216533"
       fy="47.338062"
       fx="41.706768"
       cy="47.338062"
       cx="41.706768"
       gradientTransform="matrix(-1.9089582,-1.2639863,1.2387528,-1.8708484,70.259621,166.51498)"
       gradientUnits="userSpaceOnUse"
       id="radialGradient3991-2"
       xlink:href="#linearGradient4016-45"
       inkscape:collect="always" />
    <linearGradient
       id="linearGradient4016-45">
      <stop
         id="stop4018-8"
         offset="0"
         style="stop-color:#ffffff;stop-opacity:1;" />
      <stop
         style="stop-color:#fffdff;stop-opacity:1;"
         offset="0.45679879"
         id="stop4020-8" />
      <stop
         style="stop-color:#7384bf;stop-opacity:1;"
         offset="0.63027239"
         id="stop4022-1" />
      <stop
         style="stop-color:#393939;stop-opacity:1;"
         offset="0.75237787"
         id="stop4024-2" />
      <stop
         id="stop4026-1"
         offset="1"
         style="stop-color:#000000;stop-opacity:1;" />
    </linearGradient>
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient4085"
       id="linearGradient5604"
       x1="38.899879"
       y1="39.905556"
       x2="46.964371"
       y2="31.924204"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(0.75938469,0,0,0.75938469,31.258584,1.43463)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient4085-8"
       id="linearGradient5604-9"
       x1="38.899879"
       y1="39.905556"
       x2="46.964371"
       y2="31.924204"
       gradientUnits="userSpaceOnUse" />
    <linearGradient
       id="linearGradient4085-8">
      <stop
         style="stop-color:#000000;stop-opacity:0.57358491;"
         offset="0"
         id="stop4087-2" />
      <stop
         id="stop4089-4"
         offset="0.31692868"
         style="stop-color:#2b2b2b;stop-opacity:0.59622639;" />
      <stop
         id="stop4091-5"
         offset="0.53031564"
         style="stop-color:#396f7c;stop-opacity:0.69433957;" />
      <stop
         id="stop4093-2"
         offset="0.83225203"
         style="stop-color:#393939;stop-opacity:0.32452831;" />
      <stop
         style="stop-color:#000000;stop-opacity:0.21132076;"
         offset="1"
         id="stop4095-9" />
    </linearGradient>
    <linearGradient
       y2="31.924204"
       x2="46.964371"
       y1="39.905556"
       x1="38.899879"
       gradientUnits="userSpaceOnUse"
       id="linearGradient5652"
       xlink:href="#linearGradient4085-8"
       inkscape:collect="always" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="12.515625"
     inkscape:cx="19.535581"
     inkscape:cy="32"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="false"
     inkscape:window-width="1920"
     inkscape:window-height="1018"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata3115">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-988.36218)">
    <image
       y="990.56219"
       x="0"
       id="image3046"
       xlink:href="data:image/png;base64,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 YII= "
       height="60"
       width="64"
       style="opacity:0.15" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880"
       width="4"
       height="4"
       x="0"
       y="1002.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1"
       width="4"
       height="4"
       x="0"
       y="1034.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1-7"
       width="4"
       height="4"
       x="60"
       y="1002.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-1-7-4"
       width="4"
       height="4"
       x="60"
       y="1034.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-0"
       width="24.049999"
       height="4"
       x="20"
       y="1014.3622" />
    <rect
       style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:0;stroke-opacity:1;stroke-dasharray:none"
       id="rect3880-0-9"
       width="24.049999"
       height="4"
       x="20"
       y="1022.3622" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 20.190939,1016.3912 c -2.727844,-0.2332 -4.854091,-1.6454 -5.386257,-2.463 -0.600298,-0.9223 -1.825207,-3.1963 -2.887612,-4.8923 -1.053476,-1.6818 -2.2256099,-3.3725 -3.0390361,-3.962 -0.7246268,-0.5252 -2.4672914,-0.7403 -4.9280952,-0.7085"
       id="path3909"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 60.241352,1004.3654 c -2.727903,0.2329 -4.854196,1.6447 -5.386374,2.4624 -0.600311,0.9221 -1.825246,3.1954 -2.887674,4.8912 -1.053499,1.6814 -2.225658,3.3715 -3.039102,3.9608 -0.724642,0.5252 -2.467345,0.7404 -4.928202,0.7083"
       id="path3909-4"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 60.269449,1036.3622 c -2.727844,-0.2332 -4.854091,-1.645 -5.386257,-2.4625 -0.600298,-0.9222 -1.825207,-3.1958 -2.887612,-4.8914 -1.053476,-1.6815 -2.225609,-3.3717 -3.039036,-3.9611 -0.724626,-0.5252 -2.467291,-0.7404 -4.928095,-0.7084"
       id="path3909-8"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
    <path
       style="fill:none;stroke:#000000;stroke-width:4;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"
       d="m 20.191291,1024.3359 c -2.727903,0.2328 -4.854196,1.6447 -5.386374,2.4623 -0.600311,0.9223 -1.825246,3.1957 -2.887674,4.8915 -1.053499,1.6815 -2.2256583,3.3717 -3.0391025,3.9611 -0.724642,0.5251 -2.467345,0.7404 -4.9282018,0.7083"
       id="path3909-82"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="csssc" />
  </g>
</svg>
   $ a n n o t a t i o n   c o n t e n t            @Q@         a n n o t a t i o n   n a m e            �;          i c o n   f i l e n a m e   
    � C : / U s e r s / a l v e s / D o c u m e n t s / P h o t o n i c s / C M L / l u m f o u n d r y _ t e m p l a t e / i c o n s / d i r e c t i o n a l _ c o u p l e r . s v g   & r o o t   a n n o t a t i o n   p o s                        c h i l d r e n   	               p r o p e r t i e s          " s t a t i c   p r o p e r t i e s       *    l o a d   f r o m   f i l e        $ n u m b e r   o f   f i r   t a p s           d i a g n o s t i c   s i z e          $ f i l t e r   f i t   r o l l o f f    ?�������   ( f i l t e r   f i t   t o l e r a n c e    ?�������    y   p o s i t i o n    @&          w i n d o w   f u n c t i o n       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                       c h o i c e s           r e c t a n g u l a r    h a m m i n g    h a n n i n g    a n n o t a t e       $ d e l a y   c o m p e n s a t i o n           > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s       2    c o n f i g u r a t i o n       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                    c h o i c e s           b i d i r e c t i o n a l    s   p a r a m e t e r s   , i n i t i a l i z e   f i l t e r   t a p s        * s   p a r a m e t e r s   f i l e n a m e   
    � C : / U s e r s / a l v e s / D o c u m e n t s / P h o t o n i c s / C M L / d c _ g a p = 2 0 0 n m _ L c = 3 7 . 5 u m . t x t    l o c a l   p a t h   
 ����    r e c i p r o c i t y       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                       c h o i c e s           i g n o r e    t e s t    e n f o r c e   2 r e m o v e   d i s c o n n e c t e d   p o r t s          f r a c t i o n a l   d e l a y        d e s c r i p t i o n   
    D O p t i c a l   N   p o r t   s - p a r a m e t e r   e l e m e n t    t y p e   
    4 O p t i c a l   N   P o r t   S - P a r a m e t e r    e n a b l e d        p r e f i x   
     S P A R    t e m p e r a t u r e    @r�        & p a s s i v i t y   t o l e r a n c e    >������   $ h o r i z o n t a l   f l i p p e d        
 o r d e r          4 m a x i m u m   n u m b e r   o f   i i r   t a p s          2 n u m b e r   o f   t a p s   e s t i m a t i o n       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                       c h o i c e s           d i s a b l e d    f i t   t o l e r a n c e    g r o u p   d e l a y   
 m o d e l   
 ����    x   p o s i t i o n    �@          p a s s i v i t y       ComboChoice       
 v a l u e            a c t i v e       lumatrix__matrixi                                          c h o i c e s           i g n o r e    t e s t    e n f o r c e    o p t i m a l     v e r t i c a l   f l i p p e d         k e y w o r d s   
    * o p t i c a l , b i d i r e c t i o n a l    l i b r a r y   
 ����    x   c o o r d i n a t e       lumatrix__matrixd        ����                                 n a m e   
     S P A R _ 1   & d i g i t a l   f i l t e r   t y p e       ComboChoice       
 v a l u e           a c t i v e       lumatrix__matrixi                                       c h o i c e s           s i n g l e   t a p    F I R    I I R   $ n u m b e r   o f   i i r   t a p s           r o t a t e d            r u n   d i a g n o s t i c         y   c o o r d i n a t e       lumatrix__matrixd        ����                                 u r l   
 ����   4 m a x i m u m   n u m b e r   o f   f i r   t a p s          $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s       *    l o a d   f r o m   f i l e           $ n u m b e r   o f   f i r   t a p s            d i a g n o s t i c   s i z e          $ f i l t e r   f i t   r o l l o f f          ( f i l t e r   f i t   t o l e r a n c e            y   p o s i t i o n           w i n d o w   f u n c t i o n            a n n o t a t e           $ d e l a y   c o m p e n s a t i o n           > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s           c o n f i g u r a t i o n           , i n i t i a l i z e   f i l t e r   t a p s           * s   p a r a m e t e r s   f i l e n a m e           l o c a l   p a t h           r e c i p r o c i t y           2 r e m o v e   d i s c o n n e c t e d   p o r t s             f r a c t i o n a l   d e l a y            d e s c r i p t i o n            t y p e           e n a b l e d            p r e f i x            t e m p e r a t u r e          & p a s s i v i t y   t o l e r a n c e          $ h o r i z o n t a l   f l i p p e d          
 o r d e r           4 m a x i m u m   n u m b e r   o f   i i r   t a p s          2 n u m b e r   o f   t a p s   e s t i m a t i o n           
 m o d e l            x   p o s i t i o n           p a s s i v i t y             v e r t i c a l   f l i p p e d           k e y w o r d s           l i b r a r y            x   c o o r d i n a t e           n a m e           & d i g i t a l   f i l t e r   t y p e           $ n u m b e r   o f   i i r   t a p s           r o t a t e d           r u n   d i a g n o s t i c            y   c o o r d i n a t e           u r l           4 m a x i m u m   n u m b e r   o f   f i r   t a p s           
 k i n d s       *    l o a d   f r o m   f i l e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ n u m b e r   o f   f i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    d i a g n o s t i c   s i z e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   $ f i l t e r   f i t   r o l l o f f       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   ( f i l t e r   f i t   t o l e r a n c e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    w i n d o w   f u n c t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ d e l a y   c o m p e n s a t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    c o n f i g u r a t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   , i n i t i a l i z e   f i l t e r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   * s   p a r a m e t e r s   f i l e n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r e c i p r o c i t y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   2 r e m o v e   d i s c o n n e c t e d   p o r t s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     f r a c t i o n a l   d e l a y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t e m p e r a t u r e       LumQuantityKind        u n i t   
     K    s t a n d a r d   u n i t   
     K    k i n d   
     T e m p e r a t u r e   & p a s s i v i t y   t o l e r a n c e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   
 o r d e r       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   4 m a x i m u m   n u m b e r   o f   i i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s   2 n u m b e r   o f   t a p s   e s t i m a t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p a s s i v i t y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   & d i g i t a l   f i l t e r   t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ n u m b e r   o f   i i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r u n   d i a g n o s t i c       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   4 m a x i m u m   n u m b e r   o f   f i r   t a p s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     D i m e n s i o n l e s s    p r i o r i t i e s       *    l o a d   f r o m   f i l e       �   $ n u m b e r   o f   f i r   t a p s      T    d i a g n o s t i c   s i z e      �   $ f i l t e r   f i t   r o l l o f f      @   ( f i l t e r   f i t   t o l e r a n c e      ,    y   p o s i t i o n       �    w i n d o w   f u n c t i o n      J    a n n o t a t e       2   $ d e l a y   c o m p e n s a t i o n      �   > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s      6    c o n f i g u r a t i o n       �   , i n i t i a l i z e   f i l t e r   t a p s      |   * s   p a r a m e t e r s   f i l e n a m e       �    l o c a l   p a t h       x    r e c i p r o c i t y      �   2 r e m o v e   d i s c o n n e c t e d   p o r t s           f r a c t i o n a l   d e l a y      �    d e s c r i p t i o n       P    t y p e       F    e n a b l e d       <    p r e f i x       Z    t e m p e r a t u r e       �   & p a s s i v i t y   t o l e r a n c e      �   $ h o r i z o n t a l   f l i p p e d       �   
 o r d e r         4 m a x i m u m   n u m b e r   o f   i i r   t a p s      r   2 n u m b e r   o f   t a p s   e s t i m a t i o n      "   
 m o d e l       d    x   p o s i t i o n       �    p a s s i v i t y      �     v e r t i c a l   f l i p p e d       �    k e y w o r d s       �    l i b r a r y       n    x   c o o r d i n a t e       �    n a m e       (   & d i g i t a l   f i l t e r   t y p e         $ n u m b e r   o f   i i r   t a p s      h    r o t a t e d       �    r u n   d i a g n o s t i c      �    y   c o o r d i n a t e       �    u r l       �   4 m a x i m u m   n u m b e r   o f   f i r   t a p s      ^    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s       *    l o a d   f r o m   f i l e   
        $ n u m b e r   o f   f i r   t a p s   
         d i a g n o s t i c   s i z e   
        $ f i l t e r   f i t   r o l l o f f   
        ( f i l t e r   f i t   t o l e r a n c e   
         y   p o s i t i o n   
         w i n d o w   f u n c t i o n   
         a n n o t a t e   
        $ d e l a y   c o m p e n s a t i o n   
        > f i l t e r   f i t   n u m b e r   o f   i t e r a t i o n s   
         c o n f i g u r a t i o n   
        , i n i t i a l i z e   f i l t e r   t a p s   
        * s   p a r a m e t e r s   f i l e n a m e   
         l o c a l   p a t h   
         r e c i p r o c i t y   
        2 r e m o v e   d i s c o n n e c t e d   p o r t s   
          f r a c t i o n a l   d e l a y   
         d e s c r i p t i o n   
         t y p e   
         e n a b l e d   
         p r e f i x   
         t e m p e r a t u r e   
     % t e m p e r a t u r e %   & p a s s i v i t y   t o l e r a n c e   
        $ h o r i z o n t a l   f l i p p e d   
        
 o r d e r   
        4 m a x i m u m   n u m b e r   o f   i i r   t a p s   
        2 n u m b e r   o f   t a p s   e s t i m a t i o n   
        
 m o d e l   
         x   p o s i t i o n   
         p a s s i v i t y   
          v e r t i c a l   f l i p p e d   
         k e y w o r d s   
         l i b r a r y   
         x   c o o r d i n a t e   
         n a m e   
        & d i g i t a l   f i l t e r   t y p e   
        $ n u m b e r   o f   i i r   t a p s   
         r o t a t e d   
         r u n   d i a g n o s t i c   
         y   c o o r d i n a t e   
         u r l   
        4 m a x i m u m   n u m b e r   o f   f i r   t a p s   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s          0 N u m e r i c a l / D i g i t a l   F i l t e r    T h e r m a l    G e n e r a l    D i a g n o s t i c    S t a n d a r d    N u m e r i c a l   $ p r i v a t e   p r o p e r t i e s          " t o u c h s t o n e   f o r m a t        " f i l e   s   p a r a m e t e r s           m           C L A S S _ V E R S I O N          
 p o r t s   	         n           s p a r a m e t e r   	               p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B+�&K3�?|p�B��?·K��?z�l�З?4���2�?�!��*�?�Ct�i�?3pc���?�<�\��?��t㶥�?k�L����?����E�q?���ø�?���%�7�?��"o.�?K�q��h?kC�=�̃?���|Έ?��&���?���DE�?�;8��=�?ÓQ�yy@W���=@���1d@����L@(&G���@m��9�-	@f����@! ��#@B���c@�Tz�j�@�g�0�@��AN�@m.]��?�b_�@4"��@M_�B�?�l,��?�rwS`f�?�Q����?ʅe�7��?h	HR�?           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���Bs�!���?�Ǆ;;�?�#����?v�;���?�7�6�N�?��;��?��v鯼�?��zv�?���?7=��v�?�;6є��?�Hk�4�?£��ԓ�?��كQ��?����cP�?nK�S��?���u
�?���=e�?� M���?�؊h��?O�/�m�?�a|W��?O�;�?��@Z��[��@r�ʑ@��2Xk@���(!@��z�!$@I"�[� '@3�V�%*@�؝	�0-@X�s�� 0@�2<B%�1@X}�$^:3@~�m�4@�I|�R_6@[��	�7@ �bk��9@�r�3�+;@�U�l��<@-?cZ'l>@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B=GA��?-LK���?�VE�f��?�6>mw��?��x[Z��?��?����?ހ�:5��?�?�T��?��׃��?�74m]�?x\=5�?�n����?�̻��?�ߊ�ۣ�?$��4l�?W�n�:1�?˭��
��?$Y�"��?�p��n�?�T��~)�?������?q���������Ɲ�׿.;~��Y�?��~Gx�?�Ζ�Rw@X*���@I�o��@����T@!m��@��K��!@z�2f$@.��O��&@�/�zR)@���"�+@=����R.@�_�$m0@?¸U�1@�����2@_�]�F4@̰�<�5@�l�y�6@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B���G�jV?h@�I��T?��& �FQ?U�6�f�G?���~�}:?)�ƔуC?U !!�Q?�_�

�Y?���E�`?�4o�)fb?Dԙqsd?��Xf?�f �'g?���<�e?�9>��b?�� �O_?��a�[?�a��VY?9���RT?��!��I?#��}�<?�ל�=�?&x���g�?_9���@,=�J�@ȱ�K�=@���OT @*.��j�@�>+�@�o��� @If�OAB#@
����&@]����(@
@�<�+@��X�<.@�����0@"T��1@B��+�3@�Q���4@�3N/sX6@�?��7@�q�I��9@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B��̉��?mTc ;;�?Z*Z���?�w����?���N�?.)���?�H���?�,W�v�?�����?��<Эv�?X�m����? /.�4�?	l�ԓ�?���Q��?t�ncP�?��p��?����t
�?z�5�<e�?<����?�q���?����m�?���uW��a8֭�;�?r͠�@X�pp��@�B��ʑ@ �RXk@�� �(!@�j���!$@j�"f� '@ix�%*@����0-@V
��� 0@��-C%�1@Ν�(^:3@ �m�4@܈ХR_6@s]I�	�7@1m��9@"A5�+;@�Tn��<@t['l>@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B=a��8�?����H��?N�����?^F��З?¨��8�?��n|�*�?Pa|�i�?[�l+��?b��z��?��	>Υ�?e�_~��?ĭ��w�q?�2z����?�Y�H�7�?�>71m.�?�K�p�h?�jJݥ̃??��}Έ?>��ѫ�?��}VE�?�l��>�?5Sl�uy@��W�:@��\�-d@Wwy��L@����|�@�蝚�-	@ �y���@�� @�ۙ��c@/�}]j�@�q��-�@��Uk�@��x��?�ya�[�@G��@(��B�?,:�$��?�EJZf�?�:����?|"Xz.��?�1�FN�?           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B8�(�H8V?q����T?!Ŧ>��Q?j�3�H?.��u�M<?�u��C�D?��1�XR?��G�0�Y?�A�[��_?]z��"b?b�ό�d?�	��f?Ԅ��ag?�$�v��e?oE���tb?�gs�-"_?[x<[\?w6g�վY?^*��]T?~[=LH?��=?~���2�?�#k7���?@���@��4�i�@[3�U@\ r �@N�^o�@G�!_�@���mI� @̢(�BI#@��R&�&@��j���(@I�o�+@f�H�b4.@4�ע�0@b�v�J2@��K؈3@R���^�4@g��EN6@Ia���7@�\k:�9@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   u p p e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�yAȝ��?PQCj��?-��!��?#T�D^��?���f��?��� ��?��^mr��?3g�٢��?ݣmC�?ۆCj�[�?�3�i3�?�5NT
�?�Є׉��?2l�@��?�>�j�?*��t/�?��4�&��??W�j��?��0o�l�?O)e@'�? l!X'��?$`z����*�|^��׿A��W�?fR��Jw�?��
�v@�LX?��@����@���T@��	�@g�֬��!@)T�u(f$@�Q�O��&@1[�prR)@P���+@�XX�R.@�� m0@�kkQ�1@�ۋ`��2@�Լ��F4@*�Uc��5@��s�6@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���BR�^n��?������?�/V\���?{V���?�k���?��rrQ��?���E��?�NPm���?$��
	�?�Lw��Z�?>�]�2�?�}b��?��D��?��a���?�|�i�?�cr��.�?���#���?��㏤��?�Î�l�?F���'�?�K�HT��?7o����~�;�u�׿K�P?[�?Qs�y�?�3�w@"�9�@v�=��@=0��T@�I��@�m�m��!@�8f$@'�`���&@@�5~R)@z�U%�+@'( �R.@]��;m0@�7��S�1@�Xې��2@�7B�F4@��J3�5@	�\x�6@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�	��'TJ?w�?�kZ?�Z��}�_?��|��U?Y)��-=?�:�oL~X?�ϟ��`?�U��#^?�<��U�S?��Cߵ�U?�}1�'�c?��h�xk?�u�X�m?�����mj?^뿐��c?���@�\?�g�kE�X?�,%hDW?���b �R?F[c�C?i��>C3?�P�C��޿��KC��?�|�E��@T�g@s�M�&p�?(�)�~@1��)��@Z�"5@Ew61�!@��z�"@[�c<J%@!�25 V(@����u+@UeCCD�.@�d5�!�0@����32@݊���3@EK�^�5@}I��Ǆ6@C���A8@Y#�;@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B1�*�J�?ȑQ���?<�=5>��?|8�m�ۗ?*��~w�?O��۬��?����L�?�o#}R��?���d�?��R�q�?�� a�?F�2�Ӊ?��V�̴�?���X1�?�X!�u:�?�8�%��?�3�#<�?ph
�΅?��G��]�?��MLq�?��r�6�?W츲�@��vS��@��O�@1��@�눌�@�9q�U@�����@U��1�q@]���I@�Y)OZ@���և�@u{��܂@����f�@�
J��@� �4�@e_W�\@�� rD @'��z� @�|�<���?_�t�?�]�#��?           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   2    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�D�+��? ��?P�L�k��?�����*�?�04��?;O��n�?�X\��?�)i���?S�+��A�?�h�t��?	?�X��?p�n�V�?�Ok����?����?>K+x2k�?�(]F���?���!�?[��{�?C���B��?�3ß-+�?׭���?�2q9�Q�?�w����?�}�y�?@����@o~���@3u�0��@�M�Uq@c<���@�S6�O�@�T��� @ښZ���!@��S��#@�����%@���=?v'@Z�e$&Y)@^r"�*@+@�[�6+-@*46;0/@�Bg0@諝�߁1@S�JY2@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B
�"QTJ?g��cvZ?S�K8p�_?��;͋U?�Fu�-=?��={m~X?�L��`?����#^?���/�S?�)a3��U?��?1�c?����xk?��EG`�m?$l���mj?3,P���c?�n}�+�\?́K�(�X?`��HDW?���R?��%��C?\��$B3?�@����޿���I��?|�b���@��?Hg@Q��p�?Tv{@�~@DL�ې�@�Nd7@��M0�!@�uTu�"@0ѿ�9J%@Pp�!�U(@v�ی�u+@�'D�.@2�S|!�0@`暨�32@5��3@D����5@4�Ƅ6@�gM}A8@����;@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   u p p e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�n]n��?eW����?�1^����?6}�S���?������?��[�Q��?�,2F��?�$T昞�?��o	�?�3H�Z�?{���2�?�����?�2��D��?Ce4���?�d9�i�?b��.�?p�6%���?c{����?��%�l�?�pn�'�?��}PT��?I��������Bv�׿/;?[�?^���y�?�)Өw@���%�@ ����@;���T@í��@@il��!@��E8f$@nɕ���&@��8~R)@��R%�+@����R.@u~;m0@�Z�S�1@>����2@KDB�F4@��3�5@�x�6@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   2   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�ތ9��?�6�D��?��6�Y��?��_Ż,�?L�%v���?lx�p�?=+���?�Yz��?�
���B�?n�9��?�7�y��?��:��W�?3�Cr*��?�0�R]�?�l5�l�?�w�CX��?��[�\#�?>��T}�?}K?��?`��-�?���,��?��ۃ_�?s!�=5�?�@D.{�?�@{�ŧ[�@� ����@|&胏q@Rm�6��@�=9w�@c?;�� @ۘ5���!@��[�#@ #/��%@㲤KGv'@��G|-Y)@����1@+@R���<+-@,�=(7/@�k����0@�7��1@�F�t"2@           p a r a m e t e r             o p e r a t i n g   s t a t e s       lumatrix__matrixd        ����            C L A S S _ V E R S I O N           i d e n t i f i e r           o u t p u t   
     p o r t   l o w e r   1    C L A S S _ V E R S I O N           m o d e   o u t p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1    m o d e   i n p u t          
 l a b e l   
     m o d e   1   
 o r t h o           C L A S S _ V E R S I O N           u i d   
     # 1   
 i n p u t   
     p o r t   l o w e r   1   
 d e l a y                s p a r a m e t e r       lumatrix__matrixd        ����                     �Ok1M�B*���^_�BT:�3�q�B�旹��B�r���B�L`��B��~�A��B*G�(o��BU�㌜��B����B�IU��Bշ{�$�B�S�R'�B*���9�BU��K�B�(FJ�]�B��x�p�B�`�5��B ��vb��B*�ۏ��BU5C?���B�w�����?6�m�Ϙ?ړ�Ex�?U�{ސ�?{��"͓?6<��}�?��#��?e[��b�?)��A/�?�ڱ�Ç�?�gF~2�?Ō�$+��?i%\�Ϯ�?4������?σ� f�?��1a��?$��t�?�6;R��?ܧ�~�?�ZR���?�N*�ꈅ?��p@�(&zۼ@Fuy��@@a����@�ͳ��Q@!�өD@8E�l3�@y�;�6@�����@��}�@P�nG�&@�:Y�3@𨹁�@�<�<�@q�S7ĕ@8��� @�ˀ-��?�v`��X @�͟�B�?o=r]���?��f\'�?   
 v a l i d        b o u n d i n g   r e c t                    @P      @P         $ a n n o t a t i o n   c o n t e n t            @Q@         a n n o t a t i o n   n a m e            �;          h e a d e r           d l l   i d    ^��    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   1    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�UUUUUU    t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   2    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�UUUUUU    t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   3    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�UUUUUU    t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   4    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�UUUUUU    t e r m i n a l   t y p e                  p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           n a m e   
     R E L A Y _ 1    k e y w o r d s   
 ����    l i b r a r y   
 ����   $ h o r i z o n t a l   f l i p p e d         e n a b l e d         v e r t i c a l   f l i p p e d         x   c o o r d i n a t e       lumatrix__matrixd        ����                                 y   c o o r d i n a t e       lumatrix__matrixd        ����                                
 m o d e l   
 ����    p r e f i x   
    
 R E L A Y    d e s c r i p t i o n   
    $ R e l a y   p o r t   e l e m e n t    t y p e   
    
 R e l a y    l o c a l   p a t h   
 ����    y   p o s i t i o n    @T@         r o t a t e d           u r l   
 ����    a n n o t a t e        x   p o s i t i o n    �q@        $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           n a m e            k e y w o r d s           l i b r a r y           $ h o r i z o n t a l   f l i p p e d           e n a b l e d             v e r t i c a l   f l i p p e d           x   c o o r d i n a t e           y   c o o r d i n a t e          
 m o d e l            p r e f i x            d e s c r i p t i o n            t y p e           l o c a l   p a t h           y   p o s i t i o n           r o t a t e d           u r l            a n n o t a t e            x   p o s i t i o n          
 k i n d s           n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           n a m e       (    k e y w o r d s       �    l i b r a r y       n   $ h o r i z o n t a l   f l i p p e d       �    e n a b l e d       <     v e r t i c a l   f l i p p e d       �    x   c o o r d i n a t e       �    y   c o o r d i n a t e       �   
 m o d e l       d    p r e f i x       Z    d e s c r i p t i o n       P    t y p e       F    l o c a l   p a t h       x    y   p o s i t i o n       �    r o t a t e d       �    u r l       �    a n n o t a t e       2    x   p o s i t i o n       �    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           n a m e   
         k e y w o r d s   
         l i b r a r y   
        $ h o r i z o n t a l   f l i p p e d   
         e n a b l e d   
          v e r t i c a l   f l i p p e d   
         x   c o o r d i n a t e   
         y   c o o r d i n a t e   
        
 m o d e l   
         p r e f i x   
         d e s c r i p t i o n   
         t y p e   
         l o c a l   p a t h   
         y   p o s i t i o n   
         r o t a t e d   
         u r l   
         a n n o t a t e   
         x   p o s i t i o n   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           i c o n      �<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="32px"
   height="32px"
   id="svg2987"
   version="1.1"
   inkscape:version="0.48.1 "
   sodipodi:docname="New document 1">
  <defs
     id="defs2989">
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3023"
       is_visible="true" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3019"
       is_visible="true" />
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 16 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="32 : 16 : 1"
       inkscape:persp3d-origin="16 : 10.666667 : 1"
       id="perspective2995" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="11.197802"
     inkscape:cx="2.0686946"
     inkscape:cy="16"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:grid-bbox="true"
     inkscape:document-units="px"
     inkscape:window-width="1920"
     inkscape:window-height="1030"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata2992">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     inkscape:label="Layer 1"
     inkscape:groupmode="layer">
    <text
       xml:space="preserve"
       style="font-size:18px;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:125%;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;font-family:Arial;-inkscape-font-specification:Arial"
       x="-34.381748"
       y="7.8881254"
       id="text3013"
       sodipodi:linespacing="125%"><tspan
         sodipodi:role="line"
         id="tspan3015"></tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0.17860648,0.02944006 L 32.059863,31.821394"
       id="path3017"
       inkscape:path-effect="#path-effect3019"
       inkscape:original-d="M 0.17860648,0.02944006 C 32.059863,31.821394 32.059863,31.821394 32.059863,31.821394"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,32 L 31.97056,0.1187433"
       id="path3021"
       inkscape:path-effect="#path-effect3023"
       inkscape:original-d="M 0,32 C 32.059863,0.1187433 31.97056,0.1187433 31.97056,0.1187433"
       inkscape:connector-curvature="0" />
    <rect
       style="color:#000000;fill:none;fill-opacity:0.70899467000000005;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none;stroke-dashoffset:0;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate"
       id="rect3025"
       width="31.97056"
       height="32.327774"
       x="0.26790971"
       y="0.029440064" />
  </g>
</svg>
   
 v a l i d        b o u n d i n g   r e c t                    @@      @@          d a r k   i c o n      d<?xml version="1.0" encoding="UTF-8"?>
<svg width="32px" height="32px" version="1.1" xmlns="http://www.w3.org/2000/svg">
 <text x="-34.381748" y="7.8881254" fill="#ffffff" font-family="Arial" letter-spacing="0px" word-spacing="0px" style="line-height:0%" xml:space="preserve"><tspan x="-34.381748" y="7.8881254" font-size="18px" style="line-height:1.25"> </tspan></text>
 <g fill="none" stroke="#fff">
  <path d="m0.17861 0.02944 31.881 31.792" stroke-width="1px"/>
  <path d="m0 32 31.971-31.881" stroke-width="1px"/>
  <rect x=".26791" y=".02944" width="31.971" height="32.328" color="#ffffff"/>
 </g>
</svg>
   $ a n n o t a t i o n   c o n t e n t            @B�         a n n o t a t i o n   n a m e            �;          i c o n   f i l e n a m e   
    B : / b u t t o n _ i m s / i m a g e s / p o r t r e l a y . s v g    h e a d e r           d l l   i d    ����    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   1    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           n a m e   
     R E L A Y _ 3    k e y w o r d s   
 ����    l i b r a r y   
 ����   $ h o r i z o n t a l   f l i p p e d         e n a b l e d         v e r t i c a l   f l i p p e d         x   c o o r d i n a t e       lumatrix__matrixd        ����                                 y   c o o r d i n a t e       lumatrix__matrixd        ����                                
 m o d e l   
 ����    p r e f i x   
    
 R E L A Y    d e s c r i p t i o n   
    $ R e l a y   p o r t   e l e m e n t    t y p e   
    
 R e l a y    l o c a l   p a t h   
 ����    y   p o s i t i o n    �A          r o t a t e d           u r l   
 ����    a n n o t a t e        x   p o s i t i o n    �q         $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           n a m e            k e y w o r d s           l i b r a r y           $ h o r i z o n t a l   f l i p p e d           e n a b l e d             v e r t i c a l   f l i p p e d           x   c o o r d i n a t e           y   c o o r d i n a t e          
 m o d e l            p r e f i x            d e s c r i p t i o n            t y p e           l o c a l   p a t h           y   p o s i t i o n           r o t a t e d           u r l            a n n o t a t e            x   p o s i t i o n          
 k i n d s           n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           n a m e       (    k e y w o r d s       �    l i b r a r y       n   $ h o r i z o n t a l   f l i p p e d       �    e n a b l e d       <     v e r t i c a l   f l i p p e d       �    x   c o o r d i n a t e       �    y   c o o r d i n a t e       �   
 m o d e l       d    p r e f i x       Z    d e s c r i p t i o n       P    t y p e       F    l o c a l   p a t h       x    y   p o s i t i o n       �    r o t a t e d       �    u r l       �    a n n o t a t e       2    x   p o s i t i o n       �    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           n a m e   
         k e y w o r d s   
         l i b r a r y   
        $ h o r i z o n t a l   f l i p p e d   
         e n a b l e d   
          v e r t i c a l   f l i p p e d   
         x   c o o r d i n a t e   
         y   c o o r d i n a t e   
        
 m o d e l   
         p r e f i x   
         d e s c r i p t i o n   
         t y p e   
         l o c a l   p a t h   
         y   p o s i t i o n   
         r o t a t e d   
         u r l   
         a n n o t a t e   
         x   p o s i t i o n   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           i c o n      �<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="32px"
   height="32px"
   id="svg2987"
   version="1.1"
   inkscape:version="0.48.1 "
   sodipodi:docname="New document 1">
  <defs
     id="defs2989">
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3023"
       is_visible="true" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3019"
       is_visible="true" />
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 16 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="32 : 16 : 1"
       inkscape:persp3d-origin="16 : 10.666667 : 1"
       id="perspective2995" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="11.197802"
     inkscape:cx="2.0686946"
     inkscape:cy="16"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:grid-bbox="true"
     inkscape:document-units="px"
     inkscape:window-width="1920"
     inkscape:window-height="1030"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata2992">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     inkscape:label="Layer 1"
     inkscape:groupmode="layer">
    <text
       xml:space="preserve"
       style="font-size:18px;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:125%;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;font-family:Arial;-inkscape-font-specification:Arial"
       x="-34.381748"
       y="7.8881254"
       id="text3013"
       sodipodi:linespacing="125%"><tspan
         sodipodi:role="line"
         id="tspan3015"></tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0.17860648,0.02944006 L 32.059863,31.821394"
       id="path3017"
       inkscape:path-effect="#path-effect3019"
       inkscape:original-d="M 0.17860648,0.02944006 C 32.059863,31.821394 32.059863,31.821394 32.059863,31.821394"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,32 L 31.97056,0.1187433"
       id="path3021"
       inkscape:path-effect="#path-effect3023"
       inkscape:original-d="M 0,32 C 32.059863,0.1187433 31.97056,0.1187433 31.97056,0.1187433"
       inkscape:connector-curvature="0" />
    <rect
       style="color:#000000;fill:none;fill-opacity:0.70899467000000005;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none;stroke-dashoffset:0;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate"
       id="rect3025"
       width="31.97056"
       height="32.327774"
       x="0.26790971"
       y="0.029440064" />
  </g>
</svg>
   
 v a l i d        b o u n d i n g   r e c t                    @@      @@          d a r k   i c o n      d<?xml version="1.0" encoding="UTF-8"?>
<svg width="32px" height="32px" version="1.1" xmlns="http://www.w3.org/2000/svg">
 <text x="-34.381748" y="7.8881254" fill="#ffffff" font-family="Arial" letter-spacing="0px" word-spacing="0px" style="line-height:0%" xml:space="preserve"><tspan x="-34.381748" y="7.8881254" font-size="18px" style="line-height:1.25"> </tspan></text>
 <g fill="none" stroke="#fff">
  <path d="m0.17861 0.02944 31.881 31.792" stroke-width="1px"/>
  <path d="m0 32 31.971-31.881" stroke-width="1px"/>
  <rect x=".26791" y=".02944" width="31.971" height="32.328" color="#ffffff"/>
 </g>
</svg>
   $ a n n o t a t i o n   c o n t e n t            @B�         a n n o t a t i o n   n a m e            �;          i c o n   f i l e n a m e   
    B : / b u t t o n _ i m s / i m a g e s / p o r t r e l a y . s v g    h e a d e r           d l l   i d    ����    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   3    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           n a m e   
     R E L A Y _ 2    k e y w o r d s   
 ����    l i b r a r y   
 ����   $ h o r i z o n t a l   f l i p p e d         e n a b l e d         v e r t i c a l   f l i p p e d         x   c o o r d i n a t e       lumatrix__matrixd        ����                                 y   c o o r d i n a t e       lumatrix__matrixd        ����                                
 m o d e l   
 ����    p r e f i x   
    
 R E L A Y    d e s c r i p t i o n   
    $ R e l a y   p o r t   e l e m e n t    t y p e   
    
 R e l a y    l o c a l   p a t h   
 ����    y   p o s i t i o n    @W          r o t a t e d            u r l   
 ����    a n n o t a t e        x   p o s i t i o n    @k@        $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           n a m e            k e y w o r d s           l i b r a r y           $ h o r i z o n t a l   f l i p p e d           e n a b l e d             v e r t i c a l   f l i p p e d           x   c o o r d i n a t e           y   c o o r d i n a t e          
 m o d e l            p r e f i x            d e s c r i p t i o n            t y p e           l o c a l   p a t h           y   p o s i t i o n           r o t a t e d           u r l            a n n o t a t e            x   p o s i t i o n          
 k i n d s           n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           n a m e       (    k e y w o r d s       �    l i b r a r y       n   $ h o r i z o n t a l   f l i p p e d       �    e n a b l e d       <     v e r t i c a l   f l i p p e d       �    x   c o o r d i n a t e       �    y   c o o r d i n a t e       �   
 m o d e l       d    p r e f i x       Z    d e s c r i p t i o n       P    t y p e       F    l o c a l   p a t h       x    y   p o s i t i o n       �    r o t a t e d       �    u r l       �    a n n o t a t e       2    x   p o s i t i o n       �    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           n a m e   
         k e y w o r d s   
         l i b r a r y   
        $ h o r i z o n t a l   f l i p p e d   
         e n a b l e d   
          v e r t i c a l   f l i p p e d   
         x   c o o r d i n a t e   
         y   c o o r d i n a t e   
        
 m o d e l   
         p r e f i x   
         d e s c r i p t i o n   
         t y p e   
         l o c a l   p a t h   
         y   p o s i t i o n   
         r o t a t e d   
         u r l   
         a n n o t a t e   
         x   p o s i t i o n   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           i c o n      �<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="32px"
   height="32px"
   id="svg2987"
   version="1.1"
   inkscape:version="0.48.1 "
   sodipodi:docname="New document 1">
  <defs
     id="defs2989">
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3023"
       is_visible="true" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3019"
       is_visible="true" />
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 16 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="32 : 16 : 1"
       inkscape:persp3d-origin="16 : 10.666667 : 1"
       id="perspective2995" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="11.197802"
     inkscape:cx="2.0686946"
     inkscape:cy="16"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:grid-bbox="true"
     inkscape:document-units="px"
     inkscape:window-width="1920"
     inkscape:window-height="1030"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata2992">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     inkscape:label="Layer 1"
     inkscape:groupmode="layer">
    <text
       xml:space="preserve"
       style="font-size:18px;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:125%;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;font-family:Arial;-inkscape-font-specification:Arial"
       x="-34.381748"
       y="7.8881254"
       id="text3013"
       sodipodi:linespacing="125%"><tspan
         sodipodi:role="line"
         id="tspan3015"></tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0.17860648,0.02944006 L 32.059863,31.821394"
       id="path3017"
       inkscape:path-effect="#path-effect3019"
       inkscape:original-d="M 0.17860648,0.02944006 C 32.059863,31.821394 32.059863,31.821394 32.059863,31.821394"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,32 L 31.97056,0.1187433"
       id="path3021"
       inkscape:path-effect="#path-effect3023"
       inkscape:original-d="M 0,32 C 32.059863,0.1187433 31.97056,0.1187433 31.97056,0.1187433"
       inkscape:connector-curvature="0" />
    <rect
       style="color:#000000;fill:none;fill-opacity:0.70899467000000005;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none;stroke-dashoffset:0;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate"
       id="rect3025"
       width="31.97056"
       height="32.327774"
       x="0.26790971"
       y="0.029440064" />
  </g>
</svg>
   
 v a l i d        b o u n d i n g   r e c t                    @@      @@          d a r k   i c o n      d<?xml version="1.0" encoding="UTF-8"?>
<svg width="32px" height="32px" version="1.1" xmlns="http://www.w3.org/2000/svg">
 <text x="-34.381748" y="7.8881254" fill="#ffffff" font-family="Arial" letter-spacing="0px" word-spacing="0px" style="line-height:0%" xml:space="preserve"><tspan x="-34.381748" y="7.8881254" font-size="18px" style="line-height:1.25"> </tspan></text>
 <g fill="none" stroke="#fff">
  <path d="m0.17861 0.02944 31.881 31.792" stroke-width="1px"/>
  <path d="m0 32 31.971-31.881" stroke-width="1px"/>
  <rect x=".26791" y=".02944" width="31.971" height="32.328" color="#ffffff"/>
 </g>
</svg>
   $ a n n o t a t i o n   c o n t e n t            @B�         a n n o t a t i o n   n a m e            �;          i c o n   f i l e n a m e   
    B : / b u t t o n _ i m s / i m a g e s / p o r t r e l a y . s v g    h e a d e r           d l l   i d    ����    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   2    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           n a m e   
     R E L A Y _ 4    k e y w o r d s   
 ����    l i b r a r y   
 ����   $ h o r i z o n t a l   f l i p p e d         e n a b l e d         v e r t i c a l   f l i p p e d         x   c o o r d i n a t e       lumatrix__matrixd        ����                                 y   c o o r d i n a t e       lumatrix__matrixd        ����                                
 m o d e l   
 ����    p r e f i x   
    
 R E L A Y    d e s c r i p t i o n   
    $ R e l a y   p o r t   e l e m e n t    t y p e   
    
 R e l a y    l o c a l   p a t h   
 ����    y   p o s i t i o n    �6          r o t a t e d            u r l   
 ����    a n n o t a t e        x   p o s i t i o n    @l@        $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           n a m e            k e y w o r d s           l i b r a r y           $ h o r i z o n t a l   f l i p p e d           e n a b l e d             v e r t i c a l   f l i p p e d           x   c o o r d i n a t e           y   c o o r d i n a t e          
 m o d e l            p r e f i x            d e s c r i p t i o n            t y p e           l o c a l   p a t h           y   p o s i t i o n           r o t a t e d           u r l            a n n o t a t e            x   p o s i t i o n          
 k i n d s           n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    k e y w o r d s       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l i b r a r y       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y   $ h o r i z o n t a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y     v e r t i c a l   f l i p p e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t    y   c o o r d i n a t e       LumQuantityKind        u n i t   
     m    s t a n d a r d   u n i t   
     m    k i n d   
     F i x e d U n i t   
 m o d e l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r e f i x       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    d e s c r i p t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    l o c a l   p a t h       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    y   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    r o t a t e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    u r l       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    x   p o s i t i o n       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           n a m e       (    k e y w o r d s       �    l i b r a r y       n   $ h o r i z o n t a l   f l i p p e d       �    e n a b l e d       <     v e r t i c a l   f l i p p e d       �    x   c o o r d i n a t e       �    y   c o o r d i n a t e       �   
 m o d e l       d    p r e f i x       Z    d e s c r i p t i o n       P    t y p e       F    l o c a l   p a t h       x    y   p o s i t i o n       �    r o t a t e d       �    u r l       �    a n n o t a t e       2    x   p o s i t i o n       �    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           n a m e   
         k e y w o r d s   
         l i b r a r y   
        $ h o r i z o n t a l   f l i p p e d   
         e n a b l e d   
          v e r t i c a l   f l i p p e d   
         x   c o o r d i n a t e   
         y   c o o r d i n a t e   
        
 m o d e l   
         p r e f i x   
         d e s c r i p t i o n   
         t y p e   
         l o c a l   p a t h   
         y   p o s i t i o n   
         r o t a t e d   
         u r l   
         a n n o t a t e   
         x   p o s i t i o n   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           i c o n      �<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="32px"
   height="32px"
   id="svg2987"
   version="1.1"
   inkscape:version="0.48.1 "
   sodipodi:docname="New document 1">
  <defs
     id="defs2989">
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3023"
       is_visible="true" />
    <inkscape:path-effect
       effect="spiro"
       id="path-effect3019"
       is_visible="true" />
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 16 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="32 : 16 : 1"
       inkscape:persp3d-origin="16 : 10.666667 : 1"
       id="perspective2995" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="11.197802"
     inkscape:cx="2.0686946"
     inkscape:cy="16"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:grid-bbox="true"
     inkscape:document-units="px"
     inkscape:window-width="1920"
     inkscape:window-height="1030"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata2992">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     id="layer1"
     inkscape:label="Layer 1"
     inkscape:groupmode="layer">
    <text
       xml:space="preserve"
       style="font-size:18px;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;line-height:125%;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;font-family:Arial;-inkscape-font-specification:Arial"
       x="-34.381748"
       y="7.8881254"
       id="text3013"
       sodipodi:linespacing="125%"><tspan
         sodipodi:role="line"
         id="tspan3015"></tspan></text>
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0.17860648,0.02944006 L 32.059863,31.821394"
       id="path3017"
       inkscape:path-effect="#path-effect3019"
       inkscape:original-d="M 0.17860648,0.02944006 C 32.059863,31.821394 32.059863,31.821394 32.059863,31.821394"
       inkscape:connector-curvature="0" />
    <path
       style="fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 0,32 L 31.97056,0.1187433"
       id="path3021"
       inkscape:path-effect="#path-effect3023"
       inkscape:original-d="M 0,32 C 32.059863,0.1187433 31.97056,0.1187433 31.97056,0.1187433"
       inkscape:connector-curvature="0" />
    <rect
       style="color:#000000;fill:none;fill-opacity:0.70899467000000005;stroke:#000000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none;stroke-dashoffset:0;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate"
       id="rect3025"
       width="31.97056"
       height="32.327774"
       x="0.26790971"
       y="0.029440064" />
  </g>
</svg>
   
 v a l i d        b o u n d i n g   r e c t                    @@      @@          d a r k   i c o n      d<?xml version="1.0" encoding="UTF-8"?>
<svg width="32px" height="32px" version="1.1" xmlns="http://www.w3.org/2000/svg">
 <text x="-34.381748" y="7.8881254" fill="#ffffff" font-family="Arial" letter-spacing="0px" word-spacing="0px" style="line-height:0%" xml:space="preserve"><tspan x="-34.381748" y="7.8881254" font-size="18px" style="line-height:1.25"> </tspan></text>
 <g fill="none" stroke="#fff">
  <path d="m0.17861 0.02944 31.881 31.792" stroke-width="1px"/>
  <path d="m0 32 31.971-31.881" stroke-width="1px"/>
  <rect x=".26791" y=".02944" width="31.971" height="32.328" color="#ffffff"/>
 </g>
</svg>
   $ a n n o t a t i o n   c o n t e n t            @B�         a n n o t a t i o n   n a m e            �;          i c o n   f i l e n a m e   
    B : / b u t t o n _ i m s / i m a g e s / p o r t r e l a y . s v g    h e a d e r           d l l   i d    ����    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   4    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e           h e a d e r           d l l   i d    ����    t y p e           r e s u l t s           r e s u l t s   	         h i s t o r y           
 p o r t s          
 p o r t s   	               p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   1    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   2    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   3    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n            p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e                  p o r t   t y p e           p r o p e r t i e s          " s t a t i c   p r o p e r t i e s           e n a b l e d        n a m e   
     p o r t   4    t y p e   
         a n n o t a t e       $ d y n a m i c   p r o p e r t i e s            m e t a   d a t a       
    o p t i o n s           e n a b l e d            n a m e            t y p e           a n n o t a t e           
 k i n d s           e n a b l e d       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    n a m e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    t y p e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    a n n o t a t e       LumQuantityKind        u n i t   
         s t a n d a r d   u n i t   
         k i n d   
     N o n Q u a n t i t y    p r i o r i t i e s           e n a b l e d           n a m e       (    t y p e           a n n o t a t e       2    c a t e g o r i e s           
 t y p e s            e x p r e s s i o n s           e n a b l e d   
         n a m e   
         t y p e   
         a n n o t a t e   
         m e t a   d a t a            l i m i t s            d e p e n d e n c i e s            a l l   c a t e g o r i e s           G e n e r a l   $ p r i v a t e   p r o p e r t i e s           p o r t   p o s i t i o n           p o r t   t y p e            p r o p e r t y   m a n a g e r            p r i o r i t y           p o r t   l o c a t i o n    ?�          t e r m i n a l   t y p e       