%function call is denoisedImage = ROF(f,tol,alpha)
%
%for inpainting set dimsU to the correct size and add
% 'operatorK' = operatorK; where K is the subsampling operator
function [result] = DenoisingFirstOrder(f,dimsU,tol,alpha,varargin)
    vararginParser;

    if(~exist('dataterm','var'))
        dataterm = '';
    end
    if(~exist('regularizer','var'))
        regularizer = '';
    end
    
    %Choose Dataterm
    if (strcmp(dataterm,'L1'))
        data = 1; %L1
    else
        data = 2; %L2
    end

    %Choose Regularizer
    if (strcmp(regularizer,'L2'))
        reg = 1; %L2
    else
        reg = 2; %TV
    end

    nPx = prod(dimsU);
    nDim = numel(dimsU);
    
    maxInnerIt = 5000;
    
    if (exist('operatorK','var'))
        listOperator{1} = operatorK;
        clear operatorK;
    else
        listOperator{1} = speye(nPx);
    end
    
    listOperator{2} = generateForwardGradientND( dimsU,ones(nDim,1) );
    
    %create y
    for i=1:numel(listOperator)
        y{i} = zeros(size(listOperator{i},1),1);
        Kx{i} = zeros(size(listOperator{i},1),1);
    end
    
    tau=1/(2*nDim+1);
    sigma = 0.25;
    
    x = zeros(nPx,1);

    Kty = x;
    
    iterations = 1;err=1;reverseStr='';
    while err>tol && iterations < maxInnerIt
        yOld = y;

        xOld = x;

        KtyOld = Kty;%K(y_{k})
        
        Kty = zeros(size(x));
        for i=1:numel(listOperator)
            Kty = Kty + listOperator{i}'*y{i};
        end
        
        x = x - tau*Kty;

        KxOld = Kx;
        for i=1:numel(listOperator)
            Kx{i} = listOperator{i}*x;
            
            yTilde{i} = y{i} + sigma*(2*Kx{i}-KxOld{i});
        end
    
        %regularizer
        if (reg == 1) %L2
            y{2} = 1/(1+sigma/alpha)*yTilde{2};
        elseif (reg == 2) %TV
            norm = zeros(nPx,1);
            for i=1:nDim
                norm = norm + yTilde{2}((i-1)*nPx+1 : i*nPx).^2;
            end

            y{2} = yTilde{2} ./ repmat(max(1,sqrt(norm/alpha)),nDim,1);
        end

        %data term
        if (data == 1) %L1
           y{1} = min(1,max(-1,yTilde{1} - sigma*f(:)));
        elseif (data == 2) %L2
           y{1} = 1/(1+sigma)*(yTilde{1} - sigma*f(:)); 
        end

        if (mod(iterations,100)==1)
            xDiff = x - xOld;
            for i=1:numel(listOperator)
                yDiff{i} = y{i} - yOld{i};
            end
            
            %primal residul
            p = xDiff/tau - (KtyOld-Kty);
            p=sum(abs(p(:)));
            
            d = 0;
            %dual residual
            for i=1:numel(listOperator)
                dTmp = yDiff{i}/sigma - (KxOld{i}-Kx{i});
                d=d+sum(abs(dTmp(:)));
            end
            
            
            
            err = (p+d) / nPx;
        
            reverseStr = printToCmd( reverseStr,sprintf('First Order Denoising\nIteration: #%d : Residual %f\n',iterations,err) );
        
            %figure(123);imagesc(f-reshape(x,size(f)));drawnow;
        end


        iterations = iterations + 1;
    end
    
    result = reshape(x,dimsU);
    
    %clear comand output
    printToCmd( reverseStr,'');
end
